# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
import importlib
import torch.multiprocessing as mp
import os
import sys
import re
from pathlib import Path
import uuid

# A loader is a python file with at least two functions
# - add_arguments - takes in a parser and adds any arguments needed
# - load_checkpoint - takes in the queue and parsed arguments

# A saver is similar but has save_checkpoint instead of
# load_checkpoint

# The loader and saver process are each given a queue, the loader
# should load the checkpoint and send the weights in messages in the
# following order, the saver should receive them in this order and
# save the checkpoints. A message consists of a python dictionary with
# a "name" for error checking and an entry for each tensor as
# indicated below. Note that the weight sent over the queue are the
# full model weights, nothing split.

# If the loader ever sends "exit" to the queue, that means something
# went wrong and it is exiting.

# - Metadata Namespace with the following attributes:
#     model_type - GPT, BERT, T5, etc.  (Part of protocol to allow this to be deduced later instead of given on command line)
#     num_layers - Number of transformer layers
#     hidden_size
#     seq_length
#     num_attention_heads
#     max_position_embeddings
#     tokenizer_type
#     iteration
#     params_dtype
#     bert_binary_head - Used only if model_type is BERT
#     previous_tensor_parallel_size - Optional
#     previous_pipeline_parallel_size - Optional
#     true_vocab_size
#     make_vocab_size_divisble_by
#     consumed_train_samples
#     consumed_valid_samples
# messages
# {
#   "name": "embeddings"
#   "position embeddings"
#   "word embeddings"
# }
# (for each transformer layer):
# {
#   "name": "transformer layer N"
#   "input layernorm weight"
#   "input layernorm bias"
#   "qkv weight"
#   "qkv bias"
#   "dense weight"
#   "dense bias"
#   "post layernorm weight"
#   "post layernorm bias"
#   "mlp l0 weight"
#   "mlp l0 bias"
#   "mlp l1 weight"
#   "mlp l1 bias"
# }
# {
#   "name": "final layer norm"
#   "weight"
#   "bias"
# }
# if present (i.e. for BERT):
# {
#   "name": "pooler"
#   "weight"
#   "bias"
# }
# {
#   "name": "lm head"
#   "dense weight"
#   "dense bias"
#   "layernorm weight"
#   "layernorm bias"
# }
# {
#   "name": "binary head"
#   "weight"
#   "bias"
# }
# - "done"

def safe_remove_directory(dir_path):
    dir_path = Path(dir_path)
    
    if not dir_path.is_dir():
        print(f"{dir_path} is not a directory.")
        return

    for item in dir_path.iterdir():
        if item.is_symlink():
            print(f"Unlinking symlink: {item}")
            item.unlink()   
        elif item.is_dir():
            print(f"Recursively removing directory: {item}")
            safe_remove_directory(item) 
        else:
            print(f"Deleting file: {item}")
            item.unlink()

    print(f"Removing directory: {dir_path}")
    dir_path.rmdir()  # Finally, remove the now-empty directory

def load_plugin(plugin_type, name):
    module_name = f"{plugin_type}_{name}"
    print(module_name)
    try:
        import loader_llama2_hf
    except ImportError as e:
        print("Error in loading", e)
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')



    known_args, _ = parser.parse_known_args()
    
    # symlink the load_dir to a temporary directory if provided specific checkpoint path
    # and write the iteration number to a file
    try:
        ckpt_iter = re.search(r'iter_(\d+)', [iter_string for iter_string in known_args.load_dir.split('/') if 'iter_' in iter_string][-1])
    except:
        ckpt_iter = None

    if ckpt_iter:
        try:
            print("Checkpoint iteration found in load_dir")
            iteration = int(ckpt_iter.group(1))
            tmp_dir_path = f".tmp_checkpoint_{uuid.uuid4()}"
            Path(tmp_dir_path).resolve().mkdir(parents=True, exist_ok=True)
            Path(os.path.join(tmp_dir_path, ckpt_iter.group())).resolve().symlink_to(Path(known_args.load_dir).resolve(), target_is_directory=True)
            with open(os.path.join(tmp_dir_path, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(iteration))
            print(f"Symlinked {known_args.load_dir} to {tmp_dir_path}")

            known_args.load_dir = tmp_dir_path
        except Exception as e:
            print(f"Error in creating symlink to checkpoint directory: {e}")
            print("Deleting temporary directory and exiting...")
            safe_remove_directory(tmp_dir_path)
            sys.exit(1)
    

    loader = load_plugin('loader', known_args.loader)
    saver = load_plugin('saver', known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    if ckpt_iter:
        args.load_dir = tmp_dir_path
        args.iteration = iteration
    
    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()

    # remove the temporary directory
    if ckpt_iter:
        safe_remove_directory(tmp_dir_path)        


if __name__ == '__main__':
    main()
