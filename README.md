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

### With Pip
Install Python 3.9, with:
- OpenCV
- Numpy
But then without "conda develop" you need to add src directory to PYTHONPATH by your own.

## Tests
Run
```bash
python -m pytest
```
test path is test/pytests