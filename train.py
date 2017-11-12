import math
import os
import random
import time

import tensorflow as tf

__all__ = [
    'run_sample_decode',
    'run_internal_eval', 'run_external_eval', 'run_full_eval', 'train'
]

def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model.model, model_dir, infer_sess, 'infer')

    _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                   infer_model.iterator, src_data, tgt_data,
                   infer_model.src_placeholder,
                   infer_model.batch_size_placeholder, summary_writer)

