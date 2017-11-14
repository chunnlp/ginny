import codecs
import time
import numpy as np
import tensorflow as tf

from ..utils import evaluation_utils
from ..utils import misc_utils as utils

__all__ = ['decode_and_evaluate', 'get_translation']


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eps,
                        num_translations_per_input=1,
                        decode=True):
    # Decode
    if decode:
        utils.print_out('  decoding to output {}'.format(trans_file))

        start_time = time.time()
        num_sentences = 0
        with codecs.getwriter('utf-8')(
            tf.gfile.GFile(trans_file, mode='wb')) as trans_f:
            trans_f.write('')

            num_translations_per_input = max(
                min(num_translations_per_input, beam_width), 1)
            while True:
                try:
                    ginny_outputs, _ = model.decode(sess)
                    if beam_width == 0:
                        ginny_outputs = np.expand_dims(ginny_outputs, 0)

                    batch_size = ginny_outputs.shape[1]
                    num_sentences += batch_size

                    for sent_id in range(batch_size):
                        for beam_id in range(num_translations_per_input):
                            translation = get_translation(
                                num_outputs[beam_id],
                                sent_id,
                                tgt_eos=tgt_eos,
                                subword_option=subword_option)
                            trans_f.write((translation + b'\n').decode('utf-8'))
                except tf.errors.OutOfRangeError:
                    utils.print_time(
                        '  done, num sentences {}, num translations per input {}'.format(
                            num_sentences, num_translations_per_input), start_time)
                    break

    # Evaluation
    evalutation_scores = {}
    if ref_file and tf.gfile.Exists(trans_file):
        for metric in metrics:
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                metric,
                subword_option=subword_option)
            evaluation_scores[metric] = score
            utils.print_out('  {} {}: {:.1f}'.format(metric, name, score))

    return evaluation_scores


def get_translation(ginny_outputs, sent_id, tgt_eos, subword_option):
    if tgt_eos: tgt_eos = tgt_eos.encode('utf-8')

    # Select a sentence
    output = ginny_outputs[sent_id, :].tolist()

    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    if subword_option == 'bpe':
        translation = utils.format_bpe_text(output)
    elif subword_option == 'spm':
        translation = utils.format_spm_text(output)
    else:
        translation = utils.format_text(output)

    return translation
