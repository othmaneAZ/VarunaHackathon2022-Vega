# VarunaHackathon2022-Vega
This repo contains the submission of Vega Team for Varuna Hackathon 2022 for the challenge of Crop Classification using Sentinel-2 Image Dataset.
This project uses the Open source project MMAction2 toolbox for video understanding based on PyTorch. It is a part of the OpenMMLab project.

# Dependencies installation
In order to be able to run the prediction inferencing Python script, you will need to have some dependencies installed on your machine.
* MMAction2 depends on PyTorch, MMCV, MMDetection (optional), and MMPose(optional). Below are quick steps for installation.
```
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```
For more information about this Package installation, please refer to the https://mmaction2.readthedocs.io/en/latest/install.html for more detailed instruction.

Other dependencies.
```
pip install rasterio geopandas 
```
# Usage

## First step
1) To use the data in a temporal model. Extract each patches from each image (total of 1317 patches/image) and save it using the command given below. make sure you provide the traindata.shp file and images directory to store each patches temporally.
```
python data_saving_for_mmaction.py
```

3)  Use the file **generate_dataset_as_txt.py** to make a customized dataset for the mmaction2. 
The input of this function is dir_path where you have the videos sub-dirs, and a path to the output text file.  
```
python generate_dataset_as_txt.py
```
4) Configuration file for mmaction2.

download the weights for the model from here: https://drive.google.com/file/d/1AGzLMKlnSo45Ra7aF4WFCLTP1yhM3Q8J/view?usp=sharing 
The only changes required in this file is to change the directories and annotation files mentioned in the config file with commnets. 
config file: **config_slowfast_varuna.py**

5) To test the model and output the results compatible to Varuna Heckathon. Execute the following command.
```
python test_slowfast.py
```

## Output file for the test set
**The polygon 167 has no data so we add a dummy label in the test file for that polygon**
