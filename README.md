# aivivn-vn-tones

## Generate data
Generate data for default problem translate_vndt
```bash
./gen_data.sh
```

Generate data for custom problem A
```bash
./gen_data.sh A
```
./export.sh 0,1 transformer_base transformer translate_vndt


## Train model
On problem `translate_vndt`, to train model `transformer` with hparams `transformer_base` on GPUs `0,1`
```bash
./train.sh 0,1 transformer_base transformer translate_vndt
```

## Predict
Similar to `train.sh`
```bash
./predict.sh 0,1 transformer_base-2 transformer_base transformer translate_vndt
```

The output is stored in `sub-translate_vndt-transformer-transformer_base.csv`

tensorboard --logdir /home/saplab/aivivn-vn-tones/t2t_train/translate_vndt/transformer-transformer_base-2
t2t-exporter \
> --model=transformer \
>  --hparams_set=transformer_base \
> --problem=translate_vndt\
> --data_dir='/home/saplab/aivivn-vn-tones/t2t_datagen' \
> --output_dir='/home/saplab/aivivn-vn-tones/t2t_train'

tensorflow_model_server \
  --port=9000 \
  --model_name=translate_vndt \
  --model_base_path=/home/saplab/aivivn-vn-tones/t2t_train/translate_vndt/transformer-transformer_base-2/export/1598851375

