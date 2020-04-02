# Emory_Plaquebox_Paper
Last updated: 2 April 2020

This repository contains the codeset that accompanies the journal article "Validation of machine learning models to detect amyloid pathologies across institutions" (Vizcarra et. al. 2020), published in [Acta Neuropathologica Communications](https://actaneurocomms.biomedcentral.com/).

Hardware requirements for running this code includes at least one CUDA-compatible GPU. This code has been tested with 1060GTX and Titan V GPU models (tested with 1 and 2 GPUs).

Software requirements include the installation of NVIDIA-Docker (which requires Docker and an appropriate NVIDIA-driver to be installed).

All code was tested on Ubuntu 18.04 LTS.

---

## Running environment
To make running the code easier, we developed a Docker image that can run a Jupyter notebook service. All the code provided is run in a series of Jupyter notebooks with comments to guide the user along. The Docker container run from the Docker image makes sure that the code will run the correct version of system and Python packages.

**Note:** the codeset makes use of Pytorch with GPU capabilities which uses cuda packages. It is possible that newer GPUs (future releases) will not be compatible with the cuda version in the Docker container. If you experience this, please make an issue in the repo and we will work on releasing a new version of the Docker image.

**Note 2:** Even though the Docker image helps in created a reproducible environment, variations will be observed. This is because the deep learning part of the analysis makes use of GPU's cuda library which can't be easily seeded for reproducibility. Expect variations but should be subtle variations.

In terminal:
```
$ docker run --gpus '"device=0"' -it --rm --ipc=host -p3333:8888 -v <data_path>:/mnt/Data/ -v <repo_path>:/mnt/AB_Plaque_Box/ jvizcar/ab_plaque_box:latest

# above command will put you inside the docker container in the /mnt/ dir, next line runs the Jupyter notebooke service
$ jupyter notebook
```
* if you have multiple GPUs, you can specify which once you want the Docker container to have access to by changing "device=0" to device=0,1,2...". Alternatively you can do "device=all" to simply use all available GPUs
* the Jupyter notebook service is ported to localhost:3333 by default, to change this modify the -p3333 to whichever port you want
* <data_path> should be the location where your data dir is (see below)
* <repo_path> should be the path to where this repo was locally cloned, (i.e. /home/username/Documenets/Emory_Plaque_Box/)

## Password
Jupyter notebook runs with a password by default for some security reasons - dGutmanLa8!


## Directory Set-up
You should have 2 directories.

1. the repo directory should be where you cloned this repo locally using ```$ git clone ....```
2. second location you must setup and is the directory location - note that you will need some hard-drive space for this
    * create a local directory (i.e. repo_data)
    * create a wsi dir inside the direcotry (i.e. repo_data/wsi)
    * in wsi create these 5 directories: Dataset_1a_Development_Train, Dataset_1b_Development_validation, Dataset_2_Hold-out,
    Dataset_3_CERAD-like_hold-out, Dataset_Emory_CERAD-like_hold-out (these 5 dirs will contain the image datasets)    
    * go to http://computablebrain.emory.edu:8080/#collection/5d607ae8d1dbc700dde750a7/folder/5e29ef629f68993bf1676f78
    * check the box next to AB_Slides
    * click the down arrow above AB_Slides and select "download checked resources" option. This will download a zip containing all the Emory AB WSI
    * go to https://doi.org/10.5281/zenodo.1470797 and download all associated zip files (5 of them)
    * unzip all of them to get the image svs files and move the images to their appropriate wsi dir
   
Example of dir structure:

-- wsi/

-- -- Dataset_1a_Development_Train/

-- -- Dataset_1b_Development_validation/

-- -- Dataset_2_Hold-out/

-- -- Dataset_3_CERAD-like_hold-out/

-- -- Dataset_Emory_CERAD-like_hold-out/

**Please follow this naming convention to minimize the modification required in the Jupyter notebooks. Inside each of these dirs there should only be the image svs files you downloaded.**

## Code
The code can be found in analysis_notebooks and are numbered in order to run them.

