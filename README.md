# ColourNet
Auto Colourization of Grayscale Images using Deep Convolutional Neural Networks

People colour old black and white photographs using Photoshop. This model automates the colourization process without human work or intervention. 

# Dataset and Images
LabelMe Dataset - Coast & Beach, Open Country, Forest, Street images

# Input-Output
RGB images converted to YUV format because the channels can be separated into intensity and chrominance.
Y channel - Intensity - Input
UV channels - Chrominance - Output

In case of test image:
Output input image Y is concatenated with output UV channels.
YUV image -> RGB format - final output coloured image

# Model
The breakdown of the model
1. A chopped VGG-16 Net for extracting feature layers
2. Parallel inverse convolution layers to bring those extracted layers in the same ht-wid dimension
3. Concatenating the layers to form a hypercolumn. It contains a lot of information about the input image.
4. A convolutional network taking in the hypercolumn and producing the output channels.

