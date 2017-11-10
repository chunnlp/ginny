import collections
import time
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

__all__ = [
    'get_initializer', 'get_device_str',
    'create_train_model', 'create_eval_model', 'create_infer_model',
    'create_emb_for_encoder_and_decoder', 'create_rnn_cell',
    'gradient_clip', 'create_or_load_model', 'load_model',
    'compute_perplexity'
]

def get_initializer(init_op, seed=None, init_weight=None):
    if init_op == 'uniform':
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == 'glorot_normal':
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == 'glorot_uniform':
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError('Unknown init_op {}'.format(init_op))

def get_device_str(device_id, num_gpus):
    if num_gpus == 0:
        return '/cpu:0'
    device_str_output = '/gpu:{}'.format(device_id % num_gpus)
    return device_str_output

class ExtraArgs(collections.namedtuple(
    'ExtraArgs', ('single_cell_fn', 'model_device_fn',
                  'attention_mechanism_fn'))):
    pass

class TrainModel(
    collections.namedtuple('TrainModel',
                           ('graph', 'model', 'iterator',
                            'skip_count_placeholder'))):
    pass

def create_train_model(
    model_creator, hparams, scope=None, num_workers=1,
    jobid=0, extra_args=None):
    src_file = '{}.{}'.format(hparams.train_prefix, hparams.src)
    tgt_file = '{}.{}'.format(hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    grapth = tf.Graph()

    with graph.as_default(), tf.container(scope or 'train'):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder,
            num_shards=num_workers,
            shard_index=jobid)

        model_device_fn = None
        if extra_args:
            model_device_fn = extra_args.model_device_fn
        with tf.device(model_device_fn):
            model = model_creator(
                hparams,
                iterator=iterator,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                source_vocab_table=src_vocab_table,
                target_vocab_table=tgt_vocab_table,
                scope=scope,
                extra_args=extra_args)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)

class EvalModel(
    collections.namedtuple('EvalModel',
                           ('graph', 'model', 'src_file_placeholder',
                            'tgt_file_placeholder', 'iterator'))):
    pass


