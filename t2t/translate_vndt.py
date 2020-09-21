from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from process_large import run
import re
import tensor2tensor.models
from utils import process_line
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry, metrics
# End-of-sentence marker.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      tf.config.experimental.set_memory_growth(gpus[0], True)
      tf.config.experimental.set_memory_growth(gpus[1], True)
  except RuntimeError as e:
    print(e)
EOS = text_encoder.EOS
_VNDT_TRAIN_DATASETS = [["/home/saplab/aivivn-vn-tones/input/corpus-full.tar.gz", ("train.d", "train.t")]]

_VNDT_DEV_DATASETS = [["/home/saplab/aivivn-vn-tones/input/corpus-full.tar.gz", ("dev.d", "dev.t")]]

_VNDT_LARGE_TRAIN_DATASETS = [["/home/saplab/aivivn-vn-tones/input/corpus-full.tar.gz", ("train_large.d", "train_large.t")]]

_VNDT_LARGE_DEV_DATASETS = [["/home/saplab/aivivn-vn-tones/input/corpus-full.tar.gz", ("dev_large.d", "dev_large.t")]]

import tensorflow.compat.v1 as tf
@registry.register_problem
class TranslateVndt(translate.TranslateProblem):
    @property
    def approx_vocab_size(self):
        return 2**15  # 32768

    def source_data_files(self, dataset_split):
            train = dataset_split == problem.DatasetSplit.TRAIN
            return _VNDT_TRAIN_DATASETS if train else _VNDT_DEV_DATASETS
    @property
    def is_generate_per_split(self):
		    # generate_data will shard the data into TRAIN and EVAL for us.
            return False
    @property
    def dataset_splits(self):
            return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
            }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
            }]
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
            del data_dir
            del tmp_dir
            del dataset_split
            co_dau=open('/home/saplab/aivivn-vn-tones/input/train.t','r')
            kho_dau=open('/home/saplab/aivivn-vn-tones/input/train.d','r')
            lines_kho_dau=kho_dau.readlines()
            lines_co_dau=co_dau.readlines()
            assert(len(lines_co_dau)==len(lines_kho_dau))
            for index,line in enumerate(zip(lines_co_dau,lines_kho_dau)):
                 if line[0] and line[1]:
                             yield{"inputs":line[1],"targets":line[0],}
@registry.register_problem
class TranslateVndtLarge(translate.TranslateProblem):
    @property
    def approx_vocab_size(self):
        return 2**16  # 32768

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _VNDT_LARGE_TRAIN_DATASETS if train else _VNDT_LARGE_DEV_DATASETS

    @property
    def decode_hooks(self):
        return []

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
        ]
