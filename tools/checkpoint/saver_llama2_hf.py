# Adapted from https://gist.github.com/devymex/734ff89ffb7ba047de177201ba90b3d1

import os, sys, torch, torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer
from pathlib import Path

megatron_dir = [parent for parent in Path(__file__).resolve().parents if 'Megatron-LM' in parent.name][0]
sys.path.append(str(megatron_dir))

CHECK_EQUAL_WITH_HF = '' # A pretrain directory eg. '/data/models/llama-2-hf/7b-chat'

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF saver.')
    group.add_argument('--tokenizer-dir', type=str, required=True)
    group.add_argument('--save-dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16',
                       help='Save model in which dtype')


def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}", flush=True)
        return val

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'GPT'
    mag_conf = md.checkpoint_args
    torch_dtype = torch.float32
    if mag_conf.bf16:
        assert mag_conf.fp16 == False
        torch_dtype = torch.bfloat16
    elif mag_conf.fp16:
        assert mag_conf.bf16 == False
        torch_dtype = torch.float16
    assert mag_conf.swiglu == True
    assert mag_conf.rotary_percent == 1.0
    if args.tokenizer_dir is not None:
        assert os.path.exists(args.tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        tokenizer.save_pretrained(args.save_dir)

    hf_config = LlamaConfig(
        vocab_size              = mag_conf.padded_vocab_size,
        hidden_size             = mag_conf.hidden_size,
        intermediate_size       = mag_conf.ffn_hidden_size,
        num_hidden_layers       = mag_conf.encoder_num_layers,
        num_attention_heads     = mag_conf.num_attention_heads,
        num_key_value_heads     = mag_conf.num_query_groups if mag_conf.num_query_groups > 1 else mag_conf.num_attention_heads, 
        max_position_embeddings = mag_conf.max_position_embeddings,
        rms_norm_eps            = mag_conf.norm_epsilon,
        tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,
        attention_bias          = mag_conf.add_bias_linear,
        torch_dtype             = torch_dtype,
        rope_theta              = mag_conf.rope_theta if hasattr(mag_conf, 'rope_theta') else 10000,
        model_type              = "llama",
        architectures           = ['LlamaForCausalLM'],
        transformers_version    = "4.33.1",
        )
    hf_config.save_pretrained(args.save_dir)

    state_dict = {}
    def set_hf_param(name, tensor: torch.Tensor, dtype=args.save_dtype):
        if dtype == 'bf16':
            tensor = tensor.to(torch.bfloat16)
        elif dtype == 'fp16':
            tensor = tensor.to(torch.float16)
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor

    def reshape_qkv_layer(param, num_splits, num_heads, head_size):
        input_shape = param.size()
        saved_shape = (num_heads, num_splits, head_size) + input_shape[1:]
        param = param.view(*saved_shape) 
        param = param.transpose(1, 0).contiguous()
        param = param.view(*input_shape)
        return param.contiguous()
    
    def process_group_query_qkv_layer(layer, hf_config):
        parts = mag_conf.num_attention_heads // hf_config.num_key_value_heads + 2
        shape = (-1,
            parts, 
            hf_config.hidden_size // hf_config.num_attention_heads, 
            hf_config.hidden_size)
        qkv_weight = layer.view(*shape)
        query_layer = qkv_weight[:,:-2].reshape(-1, hf_config.hidden_size)
        key_value_layer = qkv_weight[:,-2:].reshape(-1, hf_config.hidden_size)
        
        query_layer = reshape_qkv_layer(query_layer, 1, hf_config.num_attention_heads, hf_config.hidden_size // hf_config.num_attention_heads )
        key_value_layer = reshape_qkv_layer(key_value_layer, 2, hf_config.num_key_value_heads, hf_config.hidden_size // hf_config.num_attention_heads)
        key_layer, value_layer = key_value_layer.chunk(2) 

        return query_layer, key_layer, value_layer

    set_hf_param('model.embed_tokens', queue_get("embeddings")["word embeddings"])
    for i_layer in range(hf_config.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.layers.{i_layer}.'
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["post norm weight"])
        set_hf_param(suffix + 'mlp.gate_proj', message["mlp l0 weight W"])
        set_hf_param(suffix + 'mlp.up_proj', message["mlp l0 weight V"])
        qkv_weight = message["qkv weight"]
        # add case for group query attention 
        if hf_config.num_key_value_heads < hf_config.num_attention_heads:
            query_layer, key_layer, value_layer = process_group_query_qkv_layer(qkv_weight, hf_config)
        else:
            qkv_weight = qkv_weight.view(hf_config.num_attention_heads, 3, -1, hf_config.hidden_size)
            qkv_weight = qkv_weight.transpose(0, 1).reshape(3, hf_config.hidden_size, hf_config.hidden_size)
            query_layer, key_layer, value_layer = qkv_weight.chunk(3)
        
        set_hf_param(suffix + 'self_attn.q_proj', query_layer)
        set_hf_param(suffix + 'self_attn.k_proj', key_layer)
        set_hf_param(suffix + 'self_attn.v_proj', value_layer)
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    if CHECK_EQUAL_WITH_HF:
        print(f'Checking with given HF model {CHECK_EQUAL_WITH_HF}')
        ref_model = AutoModelForCausalLM.from_pretrained(CHECK_EQUAL_WITH_HF)
        ref_state_dict = ref_model.state_dict()
        assert sorted(list(ref_state_dict.keys())) == sorted(list(state_dict.keys()))
        for key in state_dict:
            assert torch.equal(ref_state_dict[key], state_dict[key])
        print(f'Check passed. {CHECK_EQUAL_WITH_HF} and {args.save_dir} are equal.')
    print("Starting to write on disk")
    torch.save(state_dict, os.path.join(args.save_dir, 'pytorch_model.bin'))
    print("Finished saving model on disk")