set -euxo pipefail
export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
N_GPUS=$[$(echo ${CUDA_VISIBLE_DEVICES} | grep -o ',' | wc -l)+1]

HPARAMS=${2:-transformer_relative}
MODEL=${3:-transformer}
PROBLEM=${4:-translate_vndt_large}
PROJECT=$(dirname ${BASH_SOURCE[0]})
T2T_CUSTOM=${PROJECT}/t2t
DATA_DIR=${PROJECT}/t2t_datagen
TRAIN_DIR=${PROJECT}/t2t_train/translate_vndt/transformer-transformer_base-2
t2t-exporter \
	--t2t_usr_dir=${T2T_CUSTOM}\
	--model=${MODEL} \
	--hparams_set=${HPARAMS} \
	--problem=${PROBLEM}\
	--data_dir=${DATA_DIR} \
	--output_dir=${TRAIN_DIR}
