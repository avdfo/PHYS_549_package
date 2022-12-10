.. highlight:: shell

============
PHYS_549_package
============
This is a Python package repository for the PHYS 549 class project, "Denoising and fine feature extraction in angle-resolved photoemission spectroscopy data using a convolutional neural network".

Directory structure
----------------------
The "PHYS_549_package" contains two separate directories, "ArpesCNN" and "SR-CNN", each of which has a separate "README" file. Please refer to each for learning about their features and functions. Such directory structure is due to the division of the task into data simulation and deep learning sectors. Note that the ArpesCNN package may not be compatible with the tensorflow environment in SR-CNN, so it is recommended to implement them in separate environments. ArpesCNN provides a package that can be imported to generate "dataset" in SR-CNN.

Operating procedure
----------------------

1. Download the "PHYS_549_package" directory or git clone in terminal::

    $ git clone https://github.com/avdfo/PHYS_549_package.git

2. Go to the terminal and create a virtual environment for the "ArpesCNN" package. For instance, if using Python 3.7:: 

    $ conda create -n ArpesCNN_sim python=3.7
    
3. Activate the virtual environment. For instance, on MacOS::

    $ conda activate ArpesCNN_sim
    
   or on Linux::
   
    $ source activate ArpesCNN_sim
    
   Then change the directory to the ArpesCNN folder::
   
    $ cd path/to/ArpesCNN
   
   Install the ArpesCNN package enabling tight-binding (TB) and angle-resolved photoemission spectroscopy (ARPES) data simulation::
   
   $ python setup.py develop
   
   The "setup.py" file will automatically install numpy, scipy, matplotlib and chinook python packages in the environment. Refer to  
   "../PHYS_549_package/ArpesCNN/CONTRIBUTING.rst" for more details regarding installing and contributing.
   
4. After installing the ArpesCNN package, run the TB and ARPES data simulation::

    $ cd path/to/PHYS_549_package/SR-CNN/dataset
    
    $ python Dataset_gen.py
    
   In "Dataset_gen.py", you can choose how many data cubes and how many slices to take within each cube.
   
   You can separately set cube numbers for training, validation, and testing datasets, but the three numbers shouold add up to the total cube number 
   specified in the "Dataset_gen.py" file. The parameters for the tight binding models are randomized.
   
   If you create the environment with python 3.9 or 3.10, an error will pop up for the "tilt.py" file in the chinook package. Go to that python file and 
   change::
   
    $ from collections import
   
   to::
   
    $ from collections.abc import
    
   Then the issue should be resolved.
   
   After running "Dataset_gen.py", 8 folders will be created under the same directory: 6 folders for the training, validation and testing datasets, 1 
   folder "Parameters" for the parameter information for each data cube, and 1 folder "Visualizations" for visualizing the simulated TB and ARPES band 
   structures.
   
   All the TB data is saved to the directories with the suffix "hr" meaning high-resolution, and all the ARPES data is saved to the directories with the 
   suffix "lr" meaning low-resolution.
   
   The naming of the ARPES (TB) cut takes the form of "ARPES_sim_x1_x2.npy" ("TB_sim_x1_x2.npy"), where "x1" is the cube index, and "x2" is the cut index 
   in the x1-th cube.
   
5. Start the deep learning sector by first deactivating the ArpesCNN environment::
    
    $ conda deactivate
    
   and::
   
    $ conda activate a_tensorflow_environment
    
   Run the SR-CNN by::
   
    $ cd path/to/PHYS_549_package/SR-CNN
    
   and::
   
    $ python run.py
    
   In "run.py", you can change the training batch size, validation batch size and epoch number. The default setting here is training batch size = 16, 
   validation batch size = 4, and epoch number = 200.
   
   Note that the ratio of your training dataset size and validation dataset size should match the ratio of training batch size and validation batch size 
   so that training and validation processes share the same times of backpropogation within each epoch. 
   
   Learning rate can be adjusted in ".../PHYS-549-package/SR-CNN/source/neuralnet.py".
   
   The "run.py" file will also perform the trained neural network on data in ".../PHYS-549-package/SR-CNN/dataset/test_exp", where it contains actual 
   ARPES experiment data.
   
6. If you decide to keep the "Checkpoint" directory in SR-CNN, the training process will start from the saved parameters provided by the checkpoint     files. Delete the Checkpoint directory to start fresh.

   The "Checkpoint" directory records the training results from a GPU training session with 200 epochs. The training is carried out on a training 
   dataset with 20480 slices generated from 320 data cubes (64 slices per cube) with training batch size = 64, validation batch size = 32 (therefore, 
   10240 slices in the validation dataset to maintain the ratio of 2:1), learning rate = 1e-5 for the last layer, and learning rate = 1e-4 for the first 
   and second layer.
   
   During generating all the training, validation and testing ARPES data, the NSR variable in "Dataset_gen.py", which controls the strength of the 
   circuit noise added to ARPES spectra, was set to 0.5.
