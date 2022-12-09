[TensorFlow] Super-Resolution CNN (SR-CNN) on ARPES data
=====

TensorFlow implementation of 'Image Super-Resolution using Deep Convolutional Network' [1] applied on angle-resolved photoemission spectroscopy (ARPES) data.

The package is adapted from <a href="https://github.com/YeongHyeon/Super-Resolution_CNN">YeongHyeon's implementation of SR-CNN</a> in order to perform denoising and fine feature extraction on angle-resolved photoemission spectroscopy (ARPES). 


## Architecture
<div align="center">
  <img src="./readme/Architecture_white.png" width="700">  
  <p>The architecture of the Super-Resolution Network (SRCNN) on ARPES data.</p>
</div>

The architecture consists of three convolutional layers with kernel sizes of 9x9, 1x1, 5x5, respectively. The simulated ARPES data is fed into the CNN as inputs, while the TB data is used as the ground truth. The cross entropy loss is calculated between the output and the ground truth, and back propogated using an Adam optimizer.   

## Results
<div align="center">
  <img src="./readme/1000.png" width="250"><img src="./readme/10000.png" width="250"><img src="./readme/100000.png" width="250">  
  <p>Reconstructed image in each iteration (1k, 10k, 100k iterations).</p>
</div>

<div align="center">
  <img src="./readme/lr.png" width="250"><img src="./readme/100000.png" width="250"><img src="./readme/hr.png" width="250">    
  <p>Comparison between the input (Bicubic Interpolated), reconstructed image (by SRCNN), and target (High-Resolution) image.</p>
</div>

## Requirements
* Python 3.6.8  
* Tensorflow 1.14.0  
* Numpy 1.14.0  
* Matplotlib 3.1.1  

## Reference
[1] Image Super-Resolution Using Deep Convolutional Networks, Chao Dong et al., https://ieeexplore.ieee.org/abstract/document/7115171/  
[2] Urban 100 dataset, Huang et al.,  https://sites.google.com/site/jbhuang0604/publications/struct_sr  
 
