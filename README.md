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
# Detach: Ctrl + Q
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
gsutil rsync -r experiments gs://tyoyo/experiments/
```
