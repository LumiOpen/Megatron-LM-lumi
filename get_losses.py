#!/usr/bin/env python3

import sys
import re

from logging import warning
from argparse import ArgumentParser


# Example Meg-DS log line:
# 511:  iteration    13910/  953674 | consumed samples:     14243840 | consumed tokens:  29171384320 | elapsed time per iteration (s): 11.18 | learning rate: 1.000E-04 | global batch size:  1024 | lm loss: 2.152785E+00 | grad norm: 0.114 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 91.621 | TFLOPs: 40.83 |

MEGDS_LOSS_RE = re.compile(r'.*?\biteration\s+(\d+).*?\bconsumed tokens:\s+(\d+).*\blm loss:\s+(\S+).*')

# Example Meg-LM log line:
# 7:  iteration        4/   30517 | consumed samples:           64 | elapsed time per iteration (ms): 15232.8 | learning rate: 6.000E-07 | global batch size:    16 | lm loss: 1.183278E+01 | loss scale: 1.0 | grad norm: 36.543 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.050 | tokens per gpu per second (tgs): 537.788 | TFLOPs: 25.55 |

MEGLM_LOSS_RE = re.compile(r'.*?\biteration\s+(\d+).*\blm loss:\s+(\S+).*')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--tokens-per-step', type=int, default=None)
    ap.add_argument('--modulo', type=int, default=None)
    ap.add_argument('logs', nargs='+')
    return ap


def loss_improvement_ratio(values):
    improved, total = 0, 0
    prev = None
    for it in sorted(values):
        tok, loss = values[it]
        if prev is not None:
            if loss < prev:
                improved += 1
            total += 1
        prev = loss
    if total == 0:
        return None
    else:
        return improved/total
    

def main(argv):
    args = argparser().parse_args(argv[1:])

    values = {}    
    for fn in args.logs:
        with open(fn) as f:
            for l in f:
                m = MEGDS_LOSS_RE.match(l)
                if m:
                    it, tok, loss = m.groups()
                    it, tok, loss = int(it), int(tok), float(loss)
                    if args.modulo is not None and it % args.modulo != 0:
                        continue
                    if it in values:
                        warning(f'overwrite it {it} {values[it]} with {(tok, loss)}')
                    values[it] = (tok, loss)
                    continue

                m = MEGLM_LOSS_RE.match(l)
                if m:
                    it, loss = m.groups()
                    it, tok, loss = int(it), None, float(loss)
                    if args.modulo is not None and it % args.modulo != 0:
                        continue
                    if it in values:
                        warning(f'overwrite it {it} {values[it]} with {(tok, loss)}')
                    values[it] = (tok, loss)
                    
    if not values:
        warning('no values found')
        return 1

    if args.tokens_per_step is not None:
        values = {
            i: (i*args.tokens_per_step, l)
            for i, (_, l) in values.items()
        }

    print('iteration\ttokens\tloss')
    for it in sorted(values):
        tok, loss = values[it]
        print(f'{it}\t{tok}\t{loss}')

    print()
    print(f'values: {len(values)}')
    print(f'loss improvement ratio: {loss_improvement_ratio(values):.1%}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
