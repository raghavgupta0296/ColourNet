# ColourNet 
Auto Colourization of Grayscale Images using Deep Convolutional Neural Networks  

People colour old black and white photographs using Photoshop. This model automates the colourization process without human intervention or work.   

# Dataset
LabelMe Dataset - Coast & Beach, Open Country, Forest, Street images; 256x256 ~1k images 

# Input-Output
RGB images converted to YUV format because the channels can be separated into intensity and chrominance </br>
Y channel - Intensity - Input  
UV channels - Chrominance - Output</br>

In case of test image: </br>
Input channel Y is concatenated with predicted output UV channels</br>
YUV image -> RGB format - final output coloured image</br>

# Model
### The breakdown of the model  
1. A chopped VGG-16 Net for extracting feature layers
2. Parallel inverse convolution layers to bring those extracted layers in the same ht-wid dimension
3. Concatenating the layers to form a hypercolumn. It contains a lot of information about the input image.
4. A convolutional network taking in the hypercolumn and producing the output channels.
5. Used Batch Normalisation and ReLu in between layers

# Error & Optimization
Euclidean distance between each pixel value in:
1. Predicted UV channels and Real Output UV channels
2. Guassian Blur of Predicted UV channels and Guassian Blur of Real Output UV channels (Kernel size 3)
3. Guassian Blur of Predicted UV channels and Guassian Blur of Real Output UV channels (Kernel size 5)</br>
Error = Average of 1,2,3

Optimizer - Adam lr=0.001 epsilon=e-08

# References
[http://tinyclouds.org/colorize/]</br>
[http://cs231n.stanford.edu/reports/2016/pdfs/205_Report.pdf]</br>
[https://arxiv.org/abs/1411.5752]

# Examples
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_30.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_30.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_1.png" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_1.png" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_2.png" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_2.png" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_3.png" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_3.png" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_5.png" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_5.png" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_6.png" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_6.png" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_7.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_7.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_8.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_8.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_19.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_19.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_20.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_20.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_21.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_21.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_23.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_23.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_24.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_24.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_29.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_29.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_13.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_13.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_18.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_18.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_22.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_22.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_25.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_25.jpg" width="256" height="256" title="Coloured Image">
</p>
<p align="center">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/original_27.jpg" width="256" height="256" title="Original Image">
  <img src="https://github.com/raghavgupta0296/ColourNet/blob/master/Sample%20Outputs/coloured_27.jpg" width="256" height="256" title="Coloured Image">
</p>
