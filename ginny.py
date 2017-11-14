import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf

FLAGS = None

def add_arguments(parser):
    parser.register('type', 'bool', lambda v: v.lower() == 'true')

    # Network's parameters
    parser.add_argument('--num_units', type=int, default=32, help='Network size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Network depth')
    parser.add_argument('--encoder_type', type=str, default='uni',
                        help='uni | bi | gnmt')
    parser.add_argument('--residual', type='bool', nargs='?', const=True,
                        default=False,
                        help='Whether to add residual connections')
    parser.add_argument('--time_major', type='bool', nargs='?', const=True,
                        help='Whether to use time-major mode for dynamic RNN')
    parser.add_argument('--num_embeddings_partitions', type=int, default=0,
                        help='Number of partitions for embedding vars')

    # Attention mechanisms
    parser.add_argument('--attention', type=str, default='',
                        help='luong | scaled_luong | bahdanau | normed_bahdanau')
    parser.add_argument(
        '--attention_architecture',
        type=str,
        default='standard',
        help='standard | gnmt | gnmt_v2'
    )
    parser.add_argument(
        '--pass_hidden_state', type='bool', nargs='?', const=True,
        default=True,
        help='Whether to pass encoder\'s hidden state to decoder when using an attention based mode'
    )

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd | adam')
    parser.add_argument('--learning_rate', type=float, default=1.,
                        help='Learning rate. Adam: 0.001 | 0.0001')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='How many steps to inverse-decay learning')
    parser.add_argument('--warmup_scheme', type=str, default='t2t',
                        help='How to warmup learning rate')
    parser.add_argument('--start_decay_step', type=int, default=0,
                        help='When to start to decay')
    parser.add_argument('--decay_steps', type=int, default=10000,
                        help='How frequent to decay')
    parser.add_argument('--decay_factor', type=float, default=1.,
                        help='How much to decay')
    parser.add_argument('--learning_rate_decay_scheme', type=str, default='',
                        help='If specified, overwrite start_decay_step, decay_steps, decay_factor')
    parser.add_argument('--num_train_steps', type=int, default=12000,
                        help='Num steps to train')
    parser.add_argument('--colocate_gradients_with_ops', type='bool', nargs='?',
                        const=True,
                        default=True,
                        help='Whether to try to colocate gradients with corresponding op')

    # Initializer
    parser.add_argument('--init_op', type=str, default='uniform',
                        help='uniform | glorot_normal | glorot_uniform')
    parser.add_argument('--init_weight', type=float, default=0.1,
                        help='For uniform init op, initialize weights between [-a, a]')

    # Data
    parser.add_argument('--src', type=str, default=None,
                        help='Source suffix, e.g., en')
    parser.add_argument('--tgt', type=str, default=None,
                        help='Target suffix, e.g., de')
    parser.add_argument('--train_prefix', type=str, default=None,
                        help='Train prefix, expect files with src/tgt suffixes')
    parser.add_argument('--dev_prefix', type=str, default=None,
                        help='Dev prefix, expect files with src/tgt suffixes')
    parser.add_argument('--test_prefix', type=str, default=None,
                        help='Test prefix, expect files with src/tgt suffixes')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Store log/model files')

    # Vocab
    parser.add_argument('--vocab_prefix', type=str, default=None,
                        help='Vocab prefix, expect files with src/tgt suffixes. ' \
                       'If None, extract from train files.')
    parser.add_argument('--sos', type=str, default='<s>',
                        help='Start-of-sentence symbol')
    parser.add_argument('--eos', type=str, default='</s>',
                        help='End-of-sentence symbol')
    parser.add_argument('--share_vocab', type='bool', nargs='?', const=True,
                        default=False,
                        help='Whether to use the source vocab and embeddings for both src and tgt')

    # Sequence lengths
    parser.add_argument('--src_max_len', type=int, default=50,
                        help='Max length of src sequences during training')
    parser.add_argument('--tgt_max_len', type=int, default=50,
                        help='Max length of tgt sequences during training')
    parser.add_argument('--src_max_len_infer', type=int, default=None,
                        help='Max length of src sequences during inference')
    parser.add_argument('--tgt_max_len_infer', type=int, default=None,
                        help='Max length of tgt sequences during inference')

    # Default setting
