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
                        help=('Vocab prefix, expect files with src/tgt suffixes. '
                              'If None, extract from train files.'))
    parser.add_argument('--sos', type=str, default='<s>',
                        help='Start-of-sentence symbol')
    parser.add_argument('--eos', type=str, default='</s>',
                        help='End-of-sentence symbol')
    parser.add_argument('--share_vocab', type='bool', nargs='?', const=True,
                        default=False,
                        help=('Whether to use the source vocab '
                              'and embeddings for both src and tgt'))

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
    parser.add_argument('--unit_type', type=str, default='lstm',
                        help='lstm | gru | layer_norm_lstm | nas')
    parser.add_argument('--forget_bias', type=str, default=1.,
                        help='Forget bias for BasicLSTMCell')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--max_gradient_norm', type=float, default=5.,
                        help='Clip gradients to this norm')
    parser.add_argument('--source_reverse', type='bool', nargs='?', const=True,
                        default=False, help='Reverse source sequence')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--steps_per_stats', type=int, default=100,
                        help=('How many training steps to do per stats logging. '
                              'Save checkpoint every 10x steps_per_stats'))
    parser.add_argument('--max_train', type=int, default=0,
                        help='Limit on the size of training_data (0: no limit)')
    parser.add_argument('--num_buckets', type=int, default=5,
                        help='Put data into similar-length buckets')

    # SPM
    parser.add_argument('--subword_option', type=str, default='',
                        choices=['', 'bpe', 'spm'],
                        help='Set to bpe or spm to activate subword desegmentations')

    # Misc
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of gpus in each worker')
    parser.add_argument('--log_device_placement', type='bool', nargs='?',
                        const=True, default=False, help='Debug GPU allocation')
    parser.add_argument('--metrics', type=str, default='bleu',
                        help=('Comma-separated list of evaluations '
                              'metrics (bleu, rouge, accuracy)'))
    parser.add_argument('--steps_per_external_eval', type=int, default=None,
                        help=('How many training steps to do per external evaluation. '
                              'Automatically set based on data if None.'))
    parser.add_argument('--scope', type=str, default=None,
                        help='Scope to put variables under')
    parser.add_argument('--hparams_path', type=str, default=None,
                        help=('Path to standard hparams json file that overrides '
                              'hparams values from FLAGS'))
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed (>0, set a specific seed)')
    parser.add_argument('--override_loaded_hparams', type='bool', nargs='?',
                        const=True, default=False,
                        help='Override loaded hparams with values specified')

    # Inference
    parser.add_argument('--ckpt', type=str, default='',
                        help='Checkpoint file to load a model for inference')
    parser.add_argument('--inference_input_file', type=str, default=None,
                        help='Set to the text to decode')
    parser.add_argument('--inference_list', type=str, default=None,
                        help=('A comma-separated list of sentence indices '
                              '(0-based) to decode'))
    parser.add_argument('--infer_batch_size', type=int, default=32,
                        help='Batch size for inference mode')
    parser.add_argument('--inference_output_file', type=str, default=None,
                        help='Output file to store decoding results')
    parser.add_argument('--inference_ref_file', type=str, default=None,
                        help='Reference file to compute evaluation scores')
    parser.add_argument('--beam_width', type=int, default=0,
                        help=('Beam width when using beam search decoder. '
                              'If 0, use standard decoder with greedy helper.'))
    parser.add_argument('--length_penalty_weight', type=float, default=0.,
                        help='Length penalty for beam search')
    parser.add_argument('--num_translations_per_input', type=int, default=1,
                        help=('Number of translations generated for each sentence. '
                              'Only used for inference'))

    # Job info
    parser.add_argument('--jobid', type=int, default=0,
                        help='Task id of the worker')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers (inference only)')

def create_hparams(flags):
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        out_dir=flags.out_dir,

        # Networks
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_paritions=flags.num_embeddings_partitions,

        # Attention mechanisms
        attention=flags.attentions,
        attention_architecture=flags.attention_architecture,
        pass_hidden_state=flags.pass_hidden_state,

        # Train
        optimizer=flag.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        start_decay_step=flag.start_decay_step,
        decay_factor=flags.decay_factor,
        decay_steps=flags.decay_steps,
        learning_rate_decay_scheme=flags.learning_rate_decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        source_reverse=flags.source_reverse,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        num_translations_per_input=flags.num_translations_per_input,

        # Vocab
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,

        # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(','),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
    )

def extend_hparams(hparams):
    # Sanity checks
    if hparams.encoder_type == 'bi' and hparams.num_layers % 2 != 0:
        raise ValueError('For bi, num_layers {} should be even'.format(hparams.num_layers))

    if (hparams.attention_architecture in ['gnmt'] and hparams.num_layers < 2):
        raise ValueError('For gnmt attention architecture, '
                         'num_layers {} should be >= 2'.format(hparams.num_layers))

    if hparams.subword_option and hparams.subword_option not in ['spm', 'bpe']:
        raise ValueError('subword option must be either spm or bpe')

    # Flags
    utils.print_out('# hparams:')
    utils.print_out('  src={}'.format(hparams.src))
    utils.print_out('  tgt={}'.format(hparams.tgt))
    utils.print_out('  train_prefix={}'.format(hparams.train_prefix))
    utils.print_out('  dev_prefix={}'.format(hparams.dev_prefix))
    utils.print_out('  test_prefix={}'.format(hparams.test_prefix))
    utils.print_out('  out_dir={}'.format(hparams.out_dir))

    # Set num_residual_layers
    if hparams.residual and hparams.num_layers > 1:
        if hparams.encoder_type == 'gnmt':
            num_residual_layers = hparams.num_layers - 2
        else:
            num_residual_layers = hparams.num_layers - 1
    else:
        num_residual_layers = 0
    hparams.add_hparam('num_residual_layers', num_residual_layers)

    ## Vocab
    # Vocab file names
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + '.' + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + '.' + hparams.tgt
    else:
        raise ValueError('hparams.vocab_prefix must be provided')

    # Source vocab
    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)

    # Target vocab
    if hparams.share_vocab:
        utils.print_out('  using source vocab for target')
        tgt_vocab_size = src_vocab_size
        tgt_vocab_file = src_vocab_file
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=hparams.check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=vocab_utils.UNK)

    hparams.add_hparam('src_vocab_size', src_vocab_size)
    hparams.add_hparam('tgt_vocab_size', tgt_vocab_size)
    hparams.add_hparam('src_vocab_file', src_vocab_file)
    hparams.add_hparam('tgt_vocab_file', tgt_vocab_file)

    # Check out dir
    if not tf.gfile.Exists(hparams.out_dir):
        utils.print_out('# Creating output directory {}...'.format(hparams.out_dir))
        tf.gfile.MakeDirs(hparams.out_dir)

    # Evaluation
    for metric in hparams.metrics:
        hparams.add_hparam('best_' + metric, 0)
        best_metric_dir = os.path.join(hparams.out_dir, 'best_' + metric)
        hparams.add_hparam('best_' + metric + '_dir', best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

    return hparams

def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
    default_hparams = utils.maybe_parse_standard_hparams(
        default_hparams, hparams_path)

    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    if default_hparams.override_loaded_hparams:
        for key in default_config:
            if getattr(hparams, key) != default_config[key]:
                utils.print_out('# Updating hparams. {}: {} -> {}'.format(
                    key, str(getattr(hparams, key)), str(default_config[key])))
                setattr(hparams, key, default_config[key])
    return hparams

def create_or_load_hparams(
    out_dir, default_hparams, hparams_path, save_hparams=True):
    # Create or load hparams from out_dir
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams, hparams_path)
        hparams = extend_hparams(hparams)
    else:
        hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)

    # Save hparams
    if save_hparams:
        utils.save_hparams(out_dir, hparams)
        for metric in hparams.metrics:
            utils.save_hparams(getattr(hparams, 'best_' + metric + '_dir'), hparams)

    # Print hparams
    utils.print_hparams(hparams)
    return hparams


