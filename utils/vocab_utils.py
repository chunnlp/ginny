import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from ..utils import misc_utils as utils


UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_ID = 0


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
    # If vocab_file doesn't exist, create from corpus_file
    if tf.gfile.Exists(vocab_file):
        utils.print_out('# Vocab file {} exists'.format(vocab_file))
        vocab = []
        with codes.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'rb')) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
        if check_special_token:
            if not unk: unk = UNK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                utils.print_out('The first 3 vocab words [{}, {}, {}]'
                                ' are not [{}, {}, {}]'.format(
                                    vocab[0], vocab[1], vocab[2],
                                    unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter('utf-8')(
                    tf.gfile.GFile(new_vocab_file, 'wb')) as f:
                    for word in vocab:
                        f.write('{}\n'.format(word))
                vocab_file = new_vocab_file
    else:
        raise ValueError('vocab_file {} does not exist'.format(vocab_file))

    vocab_size = len(vocab)
    return vocab_size, vocab_file

def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    # Create vocab tables for src_vocab_file and tgt_vocab_file
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_vale=UNK_ID)
    return src_vocab_table, tgt_vocab_table
