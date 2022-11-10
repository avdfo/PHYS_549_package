# PHYS-549-package
This is a Python package repository for the PHYS 549 class project

## Operating procedure
1. Download the "PHYS-549-package" directory.
2. Create a virtual environment to run the "setup.py" in the ".../PHYS-549-package/ArpesCNN" directory. The ArpesCNN package is then installed
for simulating tight-binding (TB) and angle-resolved photoemission spectroscopy (ARPES) data.
3. Go to "PHYS-549-package/Generate dataset" folder and run the file "Dataset_gen.py", where you can choose how many data cubes and how many slices to 
take within each cube. The parameters for the tight binding models are randomized. After running "Dataset_gen.py", four folders will be created under the same directory. The "ARPES_simulation" folder is for all the ARPES simulated cuts, while all of their TB counterparts (as ground truth) are saved in "TB_simulation" folder. The naming of the ARPES (TB) cut takes the form of "ARPES_sim_x1_x2.npy" ("TB_sim_x1_x2.npy"), where "x1" is the cube index, and "x2" is the cut index in the x1-th cube. Randomized parameters for each cube are recorded in the "Parameters" folder. "Visualizations" folder shows all the corresponding simulated TB and ARPES images.
4. With all the data generated, copy and paste the ".npy" files to ".../PHYS-549-package/SR-CNN/dataset". Put the TB data into the high resolution (hr) folders and ARPES data into the low resolution (lr) folder. Separate the whole generated dataset into training, validation and final test datasets according to your own choice. 
5. Go to a tensorflow environment to run file ".../PHYS-549-package/SR-CNN/run.py" to train the neural network. In the ".../PHYS-549-package/SR-CNN/run.py" file, you can change the batch size and epoch number. The default setting here is batch size = 16, epoch number = 20. Learning rate can be adjusted in ".../PHYS-549-package/SR-CNN/source/neuralnet.py". Note that the Tensorflow environment may not be compatible with the ArpesCNN or Chinook environment, so it is recommended to run the dataset simulation and neural network training in separate virtual environment.
