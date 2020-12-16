# spell_correction

[Visualization Demo is Here!](https://spell-correction.herokuapp.com/)


## Usage
### Using Docker
```shell
docker build . -t spell_correction

docker run --rm -it -v $PWD:/app -p 11111:11111 spell_correction bash
docker run --rm -it -v $PWD:/app -p 11111:11111 spell_correction streamlit run --server.port 11111 app.py
docker run --rm -it -v $PWD:/app -p 11111:11111 spell_correction jupyter lab --port=11111 --ip=0.0.0.0 --no-browser --allow-root
```
