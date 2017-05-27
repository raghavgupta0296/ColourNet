# ColourNet
Auto Colourization of Grayscale Images using Deep Convolutional Neural Networks  

People colour old black and white photographs using Photoshop. This model automates the colourization process without human work or intervention.   

# Dataset and Images
LabelMe Dataset - Coast & Beach, Open Country, Forest, Street images  

# Input-Output
RGB images converted to YUV format because the channels can be separated into intensity and chrominance </br>
Y channel - Intensity - Input  
UV channels - Chrominance - Output</br>

In case of test image: </br>
Output input image Y is concatenated with output UV channels</br>
YUV image -> RGB format - final output coloured image</br>

# Model
### The breakdown of the model  
1. A chopped VGG-16 Net for extracting feature layers
2. Parallel inverse convolution layers to bring those extracted layers in the same ht-wid dimension
3. Concatenating the layers to form a hypercolumn. It contains a lot of information about the input image.
4. A convolutional network taking in the hypercolumn and producing the output channels.

# Error & Optimization
Euclidean distance between each pixel value in:
1. Predicted UV channels and Real Output UV channels
2. Guassian Blur of Predicted UV channels and Guassian Blur of Real Output UV channels (Kernel size 3)
3. Guassian Blur of Predicted UV channels and Guassian Blur of Real Output UV channels (Kernel size 5)</br>
Error = Average of 1,2,3

Optimizer - Adam

# Examples
<p align="center">
  <img src="https://assets-cdn.github.com/images/modules/logos_page/GitHub-Mark.png" width="256" height="256" title="Github Logo">
  <img src="https://assets-cdn.github.com/images/modules/logos_page/GitHub-Mark.png" width="256" height="256" title="Github Logo2">
</p>
