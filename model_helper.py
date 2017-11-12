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

def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or 'eval'):
        scr_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
                src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            ramdom_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return EvalModel(
        graph=graph,
        model=model,
        src_file_placeholder=src_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder,
        iterator=iterator)

class InferModel(
    collections.namedtuple('InferModel',
                           ('graph', 'model', 'src_placeholder',
                            'batch_size_placeholder', 'iterator'))):
    pass

def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container(scope or 'infer'):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_filem tgt_vocab_file, hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocav_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(
            src_placeholder)
        iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            src_max_len=hparams.src_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_tabel,
            scope=scope,
            extra_args=extra_args)
    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)

def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0,
                                       scope=None):
    if num_partitions <= 1:
        partitioner = None
    else:
        partitioner = tf.fixed_size_partitioner(num_partitions)

    with tf.variable_scope(
        scope or 'embeddings', dtype=dtype, partitioner=partitioner) as scope:
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError('Share embedding but different src/tgt vocab sizes'
                                 '{} vs {}'.format(src_vocab_size, tgt_vocab_size))
            utils.print_out('# Use the same source embeddings for target')
            embedding = tf.get_variable(
                'embedding_share', [src_vocab_size, src_embed_size], dtype)
            embedding_encoder = embedding
            embedding_decoder = embedding
        else:
            with tf.variable_scope('encoder', partitioner=partitioner):
                embedding_encoder = tf.get_variable(
                    'embedding_encoder', [src_vocab_size, src_embed_size], dtype)

            with tf.variable_scope('decoder', partitioner=partitioner):
                embedding_decoder = tf.get_variable(
                    'embedding_decoder', [tgt_vocab_size, tgt_embed_size], dtype)

    return embedding_encoder, embedding_decoder

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.

    if unit_type == 'lstm':
        utils.print_out(' LSTM, forget_bias={}'.format(forget_bias), new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == 'gru':
        utils.print_out(' GRU', new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == 'layer_norm_lstm':
        utils.print_out(' Layer Normalized LSTM, forget_bias={}'.format(forget_bias),
                        new_line=False)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    elif unit_type == 'nas':
        utils.print_out(' NASCell', new_line=False)
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError('Unknown unit type {}'.format(unit_type))

    if dropout > 0.:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1. - dropout))
        utils.print_out(' {}, dropout={} '.format(type(single_cell).__name__, dropout),
                        new_line=False)

    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(
            single_cell, residual_fn=residual_fn)
        utils.print_out(' {}, device={}'.format(type(single_cell).__name__, device_str),
                        new_line=False)

    return single_cell

def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        utils.print_out(' cell {}'.format(i), new_line=False)
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
            residual_fn=residual_fn
        )
        utils.print_out('')
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar('grad_norm', gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar('clipped_gradient', tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary

def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    utils.print_out(
        ' loaded {} model parameters from {}, time {:.2f}s'.format(
            name, ckpt, time.time() - start_time))
    return model

def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out(' created {} model with fresh parameters, time {:.2f}s'.format(
            name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return mode, global_step

def compute_perplexity(model, sess, name):
    total_loss = 0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            loss, predict_count, batch_size = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break

    perplexity = utils.safe_exp(total_loss / total_predict_count)
    utils.print_time(' eval {}: perplexity {:.2f}'.format(name, perplexity),
                     start_time)
    return perplexity
