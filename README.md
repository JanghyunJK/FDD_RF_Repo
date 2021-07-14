# FDD_RF_Repo

This project is a work-in-progress.

## Installation

Download and install [git](https://git-scm.com/download/win)

Download and install the latest version of [Conda](https://docs.conda.io/en/latest/) (version 4.4 or above)

Run Anaconda Prompt as Administrator

Create a new conda environment:

`$ conda create -n <name-of-repository> python=3.6 pip`

`$ conda activate <name-of-repository>`

(If you’re using a version of conda older than 4.4, you may need to instead use source activate <name-of-repository>.)

Ensure that you have navigated to the top level of your cloned repository. You will execute all your pip commands from this location. For example:

`$ cd /path/to/repository`

Install the environment needed for this repository:

`$ pip install -e .[dev]`
 
 ## Download data
Download data from https://nrel.app.box.com/folder/75255495838?s=fzn8t5t72w8yvgjogs8zfyeix7djj3pi
Unzip the files and make sure the directory is in the same format as 'data/TN_Knoxville/TN_Knoxville/...' where the metadata file and simulation data are located
 
## To-do list:
  1. Add cross validation module
  2. Add systematic feature extraction and selection module
