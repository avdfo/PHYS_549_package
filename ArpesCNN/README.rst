===================
Tight-binding and ARPES band structure simulation
===================

A package that simulates tight-binding and angle-resolved photoemission spectroscopy band sturcture customized for super-resolution convolutional neural network.


* Free software: GNU General Public License v3


Features
--------

.. image:: https://github.com/avdfo/PHYS_549_package/blob/main/ArpesCNN/README/TB_vs_ARPES.png
  :width: 450
  
* Simplify defining the dictionaries fed to Chinook. A template for how to use ArpesCNN.simulation is shown in "PHYS_549_package/SR-CNN/dataset/Dataset_gen.py".

* Calculate the tight-binding (TB, left) and the corresponding angle-resolved photoemission spectroscopy (ARPES, right) band structure, as shown above.

* Neglect the matrix element effect in ARPES spectra to accelarate the data generation. However, the option of calculating matrix element effect is still included. Electron proper self-energy terms included as well.

* Three layers of noises added to the ARPES spectra to emulate real ARPES data. One layer of Poisson noise + Gaussian filter simulating finite instrumental resolution + Gaussian noise mimicking circuit noise in experiments.

* Anti-aliasing processing on calculated TB band structure. The final TB array is still normalized to 1 at pixels with bands intersecting them.

* Functions that pull the 2D slices from the 3D TB and ARPES datacube. The selected slices are saved in the ".npy" format as inputs (ARPES) and ground truth (TB) for the super-resolution convolutional neural network (SR-CNN).

* Directly save the generated ARPES and TB data to the dataset directory of the SR-CNN.

Credits
-------

* This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
* This package imports Chinook_ to complete the tasks of building up lattice, orbital, momentum path and Hamiltonian. The diagonalization of tight-binding matrices, calculation of  self energy and ARPES spectral function are based on the implementation in Chinook.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Chinook: https://chinookpy.readthedocs.io/en/latest/introduction.html
