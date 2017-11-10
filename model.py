import abc
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

__all__ = ['BaseModel', 'Model']

class BaseModel(object):
    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None,
                 extra_args=None):
        assert isinstance(iterator, iterator_utils.BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size

        self.num_layers = hparams.num_layers
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        self.single_cell_fn = None
        if extra_args:
            self.single_cell_fn = extra_args.single_cell_fn

        # Initializer
        initializer = model_helper.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.init_embeddings(hparams, scope)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        # Projection
        with tf.variable.scope(scope or 'build_network'):
            with tf.variable.scope('decoder/output_projection'):
                self.output_layer = layers_core.Dense(
                    hparams.tgt_vocab_size, use_bias=False, name='output_projection')

        # Train graph
        res = self.build_graph(hparams, scope=scope)

        if self.mode == tf.config.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(
                self.iterator.source_sequence_length) + tf.reduce_sum(
                    self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(
                tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.predict_count = tf.reduce_sum(
                self.iterator.target_sequence_length)

        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        # Gradients and SGD update
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            # Warm-up
            self.learning_rate = self._get_learning_rate_warmup(hparams)
            # Decay
            self.learning_rate = self._get_learning_rate_decay(hparams)

            # Optimizer
            if hparams.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar('lr', self.learning_rate)
            elif hparams.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradients
            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradient_with_ops=hparams.colocate_gradients_with_ops)

            clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
                gradients, max_gradient_norm=hparams.max_gradient_norm)

            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('lr', self.learning_rate),
                tf.summary.scalar('train_loss', self.train_loss),
            ] + gradient_norm_summary)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary(hparams)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        utils.print_out('# Trainable variables')
        for param in params:
            utils.print_out(' {}, {}, {}'.format(param.name,
                                                 str(param.get_shape()),
                                                 param.op.device))

    def _get_learning_rate_warmup(self, hparams):
        return

    def _get_learning_rate_decay(self, hparams):
        return

    def init_embeddings(self, hparams, scope):
        ''' Init embeddings '''
        self.embedding_encoder, self.embedding_decoder = (
            mode_helper.create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                num_partitions=hparams.num_embeddings_partitions,
                scope=scope,))

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predict_count,
                         self.batch_size])

    def build_graph(self, hparams, scope=None):
        utils.print_out('# Creating {} graph ...'.format(self.mode))
        dtype = tf.float32
        num_layers = hparams.num_layers
        num_gpus = hparams.num_gpus

        with tf.variable_scope(scope or 'dynamic_seq2seq', dtype=dtype):
            # Encoder
            encoder_outputs, encoder_state = self._build_encoder(hparams)

            # Decoder
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams)

            # Loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        pass

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                            base_gpu=0):
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
            utils.print_out(' decoding maximum_iterations {}'.format(maximum_iterations))
        else:
            decoding_length_factor = 2.
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):

