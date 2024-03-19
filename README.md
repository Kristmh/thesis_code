# Master thesis code 

## Fox setup
pip install --user -r requirements_ml_nodes.txt

add this at the start of every notebook:
import sys
sys.path.append("/fp/homes01/u01/ec-krimhau/.local/lib/python3.11/site-packages")


## Login
ssh -L 7765:localhost:7765 -J username@gothmog.uio.no username@ml3.hpc.uio.no

## Install conda 
* module load Miniconda3/4.9.2
* or install conda: https://docs.conda.io/projects/miniconda/en/latest/ 
## Create env and install packages 
conda create -n ml python=3.10
* conda activate ml
* conda install jupyterlab
* conda install pandas
* conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
* conda install -c conda-forge transformers datasets accelerate scikit-learn papermill

## Run jupyter notebook with cuda 
CUDA_VISIBLE_DEVICES=2 jupyter-lab --port=7765 --no-browser
