# ZPDS23L
Photo denoising

## Setup
### With Conda
Install conda from https://docs.conda.io/projects/conda/en/latest/user-guide/install
Then, use command 
```bash
conda env create -f env.yaml
```
Finally 
```bsh
conda develop src
```
<!-- TODO add Flask to conda install -->

### With Pip
Install Python 3.9, with:
- OpenCV
- Numpy
- Flask

But then without "conda develop" you need to add src directory to PYTHONPATH by your own like below (remember to adjust your path accordingly).
```bash
export PYTHONPATH=${PYTHONPATH}:${HOME}/ZPDS23L/src
```

## Run

1. Move to the project directory
```bash
cd ZPDS23L
```

2. Create a `data` directory for storing images
```bash
mkdir data
```

3. Run a web app by typing the following in a command line
```bash
python3 src/run.py
```

4. And click this [link](http://localhost:8000/).
**HAVE FUN**

## Tests
Run
```bash
python -m pytest
```
test path is test/pytests
