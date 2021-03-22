# spell_correction

[Visualization Demo is Here!](https://spell-correction.herokuapp.com/)


## Usage
### Using Docker (Recommended)
1. You should set up your .env file
2. Use Docker
   * To use gpus, [you should install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
      * c.f. https://medium.com/nvidiajapan/nvidia-docker-って今どうなってるの-20-09-版-558fae883f44


```shell
docker build . -t spell_correction

docker run --rm -it -v $PWD:/app -p 11111:11111 --env-file .env --gpus all spell_correction bash
docker run --rm -it -v $PWD:/app -p 11111:11111 --env-file .env --gpus all spell_correction streamlit run --server.port 11111 app.py
docker run --rm -it -v $PWD:/app -p 11111:11111 --env-file .env --gpus all spell_correction jupyter lab --port=11111 --ip=0.0.0.0 --no-browser --allow-root


docker run -d -it -v $PWD:/app --env-file .env --gpus all spell_correction bash
docker ps
docker attach [docker container id]
# Detach: Ctrl + P -> Ctrl + Q
docker start [docker container id]
```

### Local Env
1. Install the following packages:
   * Python3.7.6
   * Poetry
2. Create virtual env
``` shell
poetry install
```

3. Configure environmental variables (for training with comet)
``` shell
# See .env.example
export COMET_DISABLE_AUTO_LOGGING=1
export COMET_API_KEY=your_api_key
# For writing to Google Storage
export GOOGLE_APPLICATION_CREDENTIALS=your_api_key
```


## Experimental Results and Models
All results are automatically uploaded to Comet.ml

Also, manually uploaded to Google Storage
```
gsutil -m rsync -r experiments gs://tyoyo/experiments/

# Download
gsutil -m rsync -r gs://tyoyo/jwtd data/jwtd/
gsutil -m rsync -r gs://tyoyo/spell_correction/data/ data/
```


```
run_name="000-mt5-large-supervised"; deepspeed spell_correction/run_seq2seq.py \
   --model_name_or_path "google/mt5-large" \
   --output_dir ./experiments/$run_name \
   --logging_dir ./experiments/$run_name/logs \
   --run_name $run_name \
   --do_train \
   --do_eval \
   --do_predict \
   --evaluation_strategy "epoch" \
   --prediction_loss_only \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --learning_rate 1e-5 \
   --weight_decay 0 \
   --adam_epsilon 1e-8 \
   --num_train_epochs 20.0 \
   --lr_scheduler_type "linear" \
   --warmup_steps 100 \
   --save_total_limit 10 \
   --seed 42 \
   --fp16 \
   --load_best_model_at_end \
   --metric_for_best_model "loss" \
   --greater_is_better false \
   --deepspeed "ds_config.json" \
   --train_file "data/jwtd/v2.1/train.json" \
   --validation_file "data/jwtd/v2.1/validation.json" \
   --test_file "data/jwtd/v2.1/test.json" \
   --preprocessing_num_workers 8 \
   --max_source_length 128 \
   --max_target_length 128 \
   --ignore_pad_token_for_loss \
   --num_beams 4 \
   --src_column "pre_text" \
   --tgt_column "post_text" \
   --dropout_rate 0.1

```
