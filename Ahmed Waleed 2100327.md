# [[2024-10-27]]

## classification Methods:

### Image Processing

*Image processing is the process of manipulating digital images. See a list of image processing techniques, including image enhancement, restoration, & others.*
#### What is Image Processing?

## What is Image Processing?

Digital Image processing is the class of methods that deal with manipulating digital images through the use of computer algorithms. It is an essential preprocessing step in many applications, such as face recognition, object detection, and image compression.

Image processing is done to enhance an existing image or to sift out important information from it. This is important in several Deep Learning-based Computer Vision applications, where such preprocessing can dramatically boost the performance of a model. Manipulating images, for example, adding or removing objects to images, is another application, especially in the entertainment industry.

![](https://framerusercontent.com/images/RFqEE0VjXYLJxEWTR9lEocSJ86I.png)

_Source:_ [_Paper_](https://arxiv.org/pdf/2203.14341)

different datasets are shown below.

![](https://framerusercontent.com/images/Hf3r73x2ix9PFpIndXzpWwLc6tE.png)

_Source:_ [_Paper_](https://arxiv.org/pdf/2203.14341)

## Types of Images / How Machines “See” Images?

Digital images are interpreted as 2D or 3D matrices by a computer, where each value or pixel in the matrix represents the amplitude, known as the “intensity” of the pixel. Typically, we are used to dealing with 8-bit images, wherein the amplitude value ranges from 0 to 255.

![](https://framerusercontent.com/images/mIaA6CShtVdalchpAK1iAzYZ6g.webp)

Thus, a computer “sees” digital images as a function: _I(x, y)_ or _I(x, y, z)_, where “_I_” is the pixel intensity and _(x, y)_ or _(x, y, z)_ represent the coordinates (for binary/grayscale or RGB images respectively) of the pixel in the image.

![](https://framerusercontent.com/images/Shq6hiScI4P6SU104kBPBuQ46m8.webp)

_Convention of the coordinate system used in an image_

Computers deal with different “types” of images based on their function representations. Let us look into them next.

### 1. Binary Image
Images that have only ==two unique== values of pixel intensity- 0 (representing black) and 1 (representing white) are called binary images. Such images are generally used to highlight a discriminating portion of a colored image. For example, it is commonly used for image segmentation, as shown below.


![](https://framerusercontent.com/images/ha4wwRY6k5tafwNZ2ELifTwqQ.png)

_Source:_ [_Paper_](https://doi.org/10.1016/j.patcog.2015.03.001)

### 2. Grayscale Image

Grayscale or 8-bit images are composed of 256 unique colors, where a pixel intensity of 0 represents the black color and pixel intensity of 255 represents the white color. All the other 254 values in between are the different shades of gray.

An example of an RGB image converted to its grayscale version is shown below. Notice that the shape of the histogram remains the same for the RGB and grayscale images.

![](https://framerusercontent.com/images/wrM9nmHdlyWAQNOGUviPfu8YU.webp)

### 3. RGB Color Image

The images we are used to in the modern world are RGB or colored images which are 16-bit matrices to computers. That is, 65,536 different colors are possible for each pixel. “RGB” represents the Red, Green, and Blue “channels” of an image.

Up until now, we had images with only one channel. That is, two coordinates could have defined the location of any value of a matrix. Now, three equal-sized matrices (called channels), each having values ranging from 0 to 255, are stacked on top of each other, and thus we require three unique coordinates to specify the value of a matrix element.

Thus, a pixel in an RGB image will be of color black when the pixel value is (0, 0, 0) and white when it is (255, 255, 255). Any combination of numbers in between gives rise to all the different colors existing in nature. For example, (255, 0, 0) is the color red (since only the red channel is activated for this pixel). Similarly, (0, 255, 0) is green and (0, 0, 255) is blue.

An example of an RGB image split into its channel components is shown below. Notice that the shapes of the histograms for each of the channels are different.

![](https://framerusercontent.com/images/uC02P5fS4lg3APtw9y4yu3wmA.webp)

_Splitting of an image into its Red, Green and Blue channels_

### 4. RGBA Image

RGBA images are colored RGB images with an extra channel known as “alpha” that depicts the opacity of the RGB image. Opacity ranges from a value of 0% to 100% and is essentially a “see-through” property.

Opacity in physics depicts the amount of light that passes through an object. For instance, cellophane paper is transparent (100% opacity), frosted glass is translucent, and wood is opaque. The alpha channel in RGBA images tries to mimic this property. An example of this is shown below.

![](https://framerusercontent.com/images/pbenGmgx2EUuQXPQG4C8MftQxQ.webp)

_Example of changing the “alpha” parameter in RGBA images_

## _Phases_ of Image Processing

**<font color="#c0504d">The fundamental steps in any typical Digital Image Processing pipeline are as follows:</font>**  
### 1. Image Acquisition

The image is captured by a camera and digitized (if the camera output is not digitized automatically) using an <font color="#1f497d">analogue-to-digital</font> converter for further processing in a computer.  
  

### 2. Image Enhancement

In this step, the acquired image is manipulated to meet the requirements of the specific task for which the image will be used. Such techniques are primarily aimed at highlighting the hidden or important details in an image, like contrast and brightness adjustment, etc. Image enhancement is highly subjective in nature.  
  

### 3. Image Restoration

This step deals with improving the appearance of an image and is an objective operation since the degradation of an image can be attributed to a mathematical or probabilistic model. For example, removing noise or blur from images.


### 4. Color Image Processing

This step aims at handling the processing of colored images (16-bit RGB or RGBA images), for example, performing color correction or color modeling in images.  
  

### 5. Wavelets and Multi-Resolution Processing

Wavelets are the building blocks for representing images in various degrees of resolution. Images subdivision successively into smaller regions for data compression and for pyramidal representation.


### 6. Image Compression

For transferring images to other devices or due to computational storage constraints, images need to be compressed and cannot be kept at their original size. This is also important in displaying images over the internet; for example, on Google, a small thumbnail of an image is a highly compressed version of the original. Only when you click on the image is it shown in the original resolution. This process saves bandwidth on the servers.

### 7. Morphological Processing

Image components that are useful in the representation and description of shape need to be extracted for further processing or downstream tasks. Morphological Processing provides the tools (which are essentially mathematical operations) to accomplish this. For example, erosion and dilation operations are used to sharpen and blur the edges of objects in an image, respectively.

### 8. Image Segmentation

This step involves partitioning an image into different key parts to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation allows for computers to put attention on the more important parts of the image, discarding the rest, which enables automated systems to have improved performance.


### 9. Representation and Description

Image segmentation procedures are generally followed by this step, where the task for representation is to decide whether the segmented region should be depicted as a boundary or a complete region. Description deals with extracting attributes that result in some quantitative information of interest or are basic for differentiating one class of objects from another.


### 10. Object Detection and Recognition

After the objects are segmented from an image and the representation and description phases are complete, the automated system needs to assign a label to the object—to let the human users know what object has been detected, for example, “vehicle” or “person”, etc.


### 11. Knowledge Base

Knowledge may be as simple as the bounding box coordinates for an object of interest that has been found in the image, along with the object label assigned to it. Anything that will help in solving the problem for the specific task at hand can be encoded into the knowledge base.

![AI video object tracking example on V7 Darwin platform](https://framerusercontent.com/images/XrLLSVjM4FL9kLw0Xq2XWzbfsfE.png)

Data labeling

Speed up your data labeling <font color="#ff0000">10x with AI</font>

Annotate videos, medical imaging, and documents faster. Use AI-assisted labeling tools, follow motion, and segment objects automatically.
## Image Processing Techniques

Image processing can be used to improve the quality of an image, remove undesired objects from an image, or even create new images from scratch. For example, image processing can be used to remove the background from an image of a person, leaving only the subject in the foreground.

Image processing is a vast and complex field, with many different algorithms and techniques that can be used to achieve different results. In this section, we will focus on some of the most common image processing tasks and how they are performed.

### **Task 1: Image Enhancement**

One of the most common image processing tasks is an image enhancement, or improving the quality of an image. It has crucial applications in Computer Vision tasks, Remote Sensing, and surveillance. One common approach is adjusting the image's contrast and brightness. 

Contrast is the difference in brightness between the lightest and darkest areas of an image. By increasing the contrast, the overall brightness of an image can be increased, making it easier to see. Brightness is the overall lightness or darkness of an image. By increasing the brightness, an image can be made lighter, making it easier to see. Both contrast and brightness can be adjusted automatically by most image editing software, or they can be adjusted manually.

![](https://framerusercontent.com/images/Skx5HahIwKG6ySWS4BUvoWuakoE.webp)

However, adjusting the contrast and brightness of an image are elementary operations. Sometimes an image with perfect contrast and brightness, when upscaled, becomes blurry due to lower pixel per square inch (pixel density). To address this issue, a relatively new and much more advanced concept of Image Super-Resolution is used, wherein a high-resolution image is obtained from its low-resolution counterpart(s). Deep Learning techniques are popularly used to accomplish this.

![](https://framerusercontent.com/images/U98hjzNULJJGOvCHo3i4AGeKf1Q.webp)

For example, the earliest example of using Deep Learning to address the Super-Resolution problem is the SRCNN model, where a low-resolution image is first upscaled using traditional Bicubic Interpolation and then used as the input to a CNN model. The non-linear mapping in the CNN extracts overlapping patches from the input image, and a convolution layer is fitted over the extracted patches to obtain the reconstructed high-resolution image. The model framework is depicted visually below.

![](https://framerusercontent.com/images/xd0KdJ9igNyXGZ3535FyJvlz88.webp)

[_SRCNN_](https://link.springer.com/content/pdf/10.1007/978-3-319-10593-2_13.pdf) _model pipeline. Image by the author_

An example of the results obtained by the SRCNN model compared to its contemporaries is shown below.

![](https://framerusercontent.com/images/8zN1UbFyaszQt2E7ITsE3ttU74.png)

[_Source: Paper_](https://link.springer.com/content/pdf/10.1007/978-3-319-10593-2_13.pdf)

### **Task 2: Image Restoration**

The quality of images could degrade for several reasons, especially photos from the era when cloud storage was not so commonplace. For example, images scanned from hard copies taken with old instant cameras often acquire scratches on them.

![](https://framerusercontent.com/images/v3razobeBdX0bPMbQlumJQzMBI.webp)

_Example of an Image Restoration operation_

Image Restoration is particularly fascinating because advanced techniques in this area could potentially restore damaged historical documents. Powerful Deep Learning-based image restoration algorithms may be able to reveal large chunks of missing information from torn documents.

Image inpainting, for example, falls under this category, and it is the process of filling in the missing pixels in an image. This can be done by using a texture synthesis algorithm, which synthesizes new textures to fill in the missing pixels. However, Deep Learning-based models are the de facto choice due to their pattern recognition capabilities.  
  

![](https://framerusercontent.com/images/ndqsG4JtuS51Jk9PPPbvccxbutI.webp)

_Example of an extreme image inpainting._ [_Source_](https://github.com/rlct1/gin)

An example of an image painting framework (based on the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) autoencoder) was proposed in this [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.pdf) that uses a two-step approach to the problem: a coarse estimation step and a refinement step. The main feature of this network is the Coherent Semantic Attention (CSA) layer that fills the occluded regions in the input images through iterative optimization. The architecture of the proposed model is shown below.

![](https://framerusercontent.com/images/r6ckXxZWVJfGZ0G07SFw6k7CM.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.pdf)

Some example results obtained by the authors and other competing models are shown below.

![](https://framerusercontent.com/images/Tz76QvQSeIDKaOxFueuoILVjAM.png)

### **Task 3: Image Segmentation**

Image segmentation the process of partitioning an image into multiple segments or regions. Each segment represents a different object in the image, and image segmentation is often used as a preprocessing step for object detection.

There are many different algorithms that can be used for image segmentation, but one of the most common approaches is to use thresholding. Binary thresholding, for example, is the process of converting an image into a binary image, where each pixel is either black or white. The threshold value is chosen such that all pixels with a brightness level below the threshold are turned black, and all pixels with a brightness level above the threshold are turned white. This results in the objects in the image being segmented, as they are now represented by distinct black and white regions.

![](https://framerusercontent.com/images/5iHBUezdFBNoTI3HVOkZYqEtYU.webp)

_Example of binary thresholding, with threshold value of 127_

In multi-level thresholding, as the name suggests, different parts of an image are converted to different shades of gray depending on the number of levels. This [paper](https://doi.org/10.1016/j.knosys.2021.107468), for example, used multi-level thresholding for [medical imaging](https://www.v7labs.com/blog/medical-image-annotation-guide)—specifically for brain MRI segmentation, an example of which is shown below.  
  

![](https://framerusercontent.com/images/K1ZfM197gKnKVW6LXUjvpj0TrVw.png)

_Source:_ [_Paper_](https://doi.org/10.1016/j.knosys.2021.107468)

Modern techniques use automated image segmentation algorithms using deep learning for both binary and multi-label segmentation problems. For example, the [PFNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Camouflaged_Object_Segmentation_With_Distraction_Mining_CVPR_2021_paper.pdf) or Positioning and Focus Network is a CNN-based model that addresses the camouflaged object segmentation problem. It consists of two key modules—the positioning module (PM) designed for object detection (that mimics predators that try to identify a coarse position of the prey); and the focus module (FM) designed to perform the identification process in predation for refining the initial segmentation results by focusing on the ambiguous regions. The architecture of the PFNet model is shown below.

![](https://framerusercontent.com/images/TMmwW2W006mnGTNPfL4Zmj8bbRQ.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Camouflaged_Object_Segmentation_With_Distraction_Mining_CVPR_2021_paper.pdf)

The results obtained by the PFNet model outperformed contemporary state-of-the-art models, examples of which are shown below.

![](https://framerusercontent.com/images/vN5OJYY0Nmpqpe0gOJeQyEFk0ug.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Camouflaged_Object_Segmentation_With_Distraction_Mining_CVPR_2021_paper.pdf)

### **Task 4: Object Detection**

Object Detection is the task of identifying objects in an image and is often used in applications such as security and surveillance. Many different algorithms can be used for object detection, but the most common approach is to use Deep Learning models, specifically Convolutional Neural Networks (CNNs).

![](https://framerusercontent.com/images/TdLowgK0b4Ldeu3zzEli9T0r10.webp)

_Object Detection with V7_

CNNs are a type of Artificial Neural Network that were specifically designed for image processing tasks since the convolution operation in their core helps the computer “see” patches of an image at once instead of having to deal with one pixel at a time. CNNs trained for object detection will output [a bounding box](https://www.v7labs.com/blog/bounding-box-annotation) (as shown in the illustration above) depicting the location where the object is detected in the image along with its class label.  
  

An example of such a network is the popular [Faster R-CNN](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) (**R**egion-based **C**onvolutional **N**eural **N**etwork) model, which is an end-to-end trainable, fully convolutional network. The Faster R-CNN model alternates between fine-tuning for the region proposal task (predicting regions in the image where an object might be present) and then fine-tuning for object detection (detecting what object is present) while keeping the proposals fixed. The architecture and some examples of region proposals are shown below.

![](https://framerusercontent.com/images/UWfnLfIIoVFJRHE2P7097jBDPQ.png)

[_Source: Paper_](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

### **Task 5: Image Compression**

Image compression is the process of reducing the file size of an image while still trying to preserve the quality of the image. This is done to save storage space, especially to run Image Processing algorithms on mobile and edge devices, or to reduce the bandwidth required to transmit the image.

Traditional approaches use lossy compression algorithms, which work by reducing the quality of the image slightly in order to achieve a smaller file size. JPEG file format, for example, uses the Discrete Cosine Transform for image compression.

Modern approaches to image compression involve the use of Deep Learning for encoding images into a lower-dimensional feature space and then recovering that on the receiver’s side using a decoding network. [Such models are called autoencoders](https://www.v7labs.com/blog/autoencoders-guide), which consist of an encoding branch that learns an efficient encoding scheme and a decoder branch that tries to revive the image loss-free from the encoded features.

![](https://framerusercontent.com/images/mpPoWPNg6qYaQkoOb5pXqYVwtY.webp)

_Basic framework for autoencoder training. Image by the author_

For example, this [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf) proposed a variable rate image compression framework using a conditional autoencoder. The conditional autoencoder is conditioned on the Lagrange multiplier, i.e., the network takes the Lagrange multiplier as input and produces a latent representation whose rate depends on the input value. The authors also train the network with mixed quantization bin sizes for fine-tuning the rate of compression. Their framework is depicted below.

![](https://framerusercontent.com/images/8nztcwUUAdJITkNaFfrVfPi7GM.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf)

The authors obtained superior results compared to popular methods like JPEG, both by reducing the bits per pixel and in reconstruction quality. An example of this is shown below.

![](https://framerusercontent.com/images/uBfOYiQAh4S22eizUJTy2EOsJQ.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf)

### **Task 6: Image Manipulation**

Image manipulation is the process of altering an image to change its appearance. This may be desired for several reasons, such as removing an unwanted object from an image or adding an object that is not present in the image. Graphic designers often do this to create posters, films, etc.

An example of Image Manipulation is [Neural Style Transfer](https://www.v7labs.com/blog/neural-style-transfer), which is a technique that utilizes Deep Learning models to adapt an image to the style of another. For example, a regular image could be transferred to the style of “Starry Night” by van Gogh. Neural Style Transfer also enables [AI to generate art](https://www.v7labs.com/blog/ai-generated-art).

![](https://framerusercontent.com/images/7RmkjEVVB8oIl9xQIpDBh5Sx38.webp)

_Example of Neural Style Transfer. Image by the author_

An example of such a model is the one proposed in this [paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf) that is able to transfer arbitrary new styles in real-time (other approaches often take much longer inference times) using an autoencoder-based framework. The authors proposed an adaptive instance normalization (AdaIN) layer that adjusts the mean and variance of the content input (the image that needs to be changed) to match those of the style input (image whose style is to be adopted). The AdaIN output is then decoded back to the image space to get the final style transferred image. An overview of the framework is shown below.

![](https://framerusercontent.com/images/oAE4GAHhXX9b9XFHAs4X1ZA6w.png)

[_Source: Paper_](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

Examples of images transferred to other artistic styles are shown below and compared to existing state-of-the-art methods.

![](https://framerusercontent.com/images/IQOOKqV01QkXLnJDRVIlm7w5uw.png)

_Source:_ [_Paper_](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

### **Task 7: Image Generation**

Synthesis of new images is another important task in image processing, especially in Deep Learning algorithms which require large quantities of labeled data to train. Image generation methods typically use [Generative Adversarial Networks](https://www.v7labs.com/blog/generative-adversarial-networks-guide) (GANs) which is [another unique neural network architecture](https://www.v7labs.com/blog/neural-network-architectures-guide).

![](https://framerusercontent.com/images/y9gAy7vMuNvFHNBqse4HRp1OL4w.webp)

_General framework for GANs. Image by the author_

GANs consist of two separate models: the generator, which generates the synthetic images, and the discriminator, which tries to distinguish synthetic images from real images. The generator tries to synthesize images that look realistic to fool the discriminator, and the discriminator trains to better critique whether an image is synthetic or real. This adversarial game allows the generator to produce photo-realistic images after several iterations, which can then be used to train other Deep Learning models.

### **Task 8: Image-to-Image Translation**

Image-to-Image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. For example, a free-hand sketch can be drawn as an input to get a realistic image of the object depicted in the sketch as the output, as shown below.

![](https://framerusercontent.com/images/XBrSQKX4A2OtPNcEEWswwBBfjAY.webp)

_Example of image-to-image translation_

‍[Pix2pix](https://arxiv.org/pdf/1611.07004.pdf) is a popular model in this domain that uses a conditional GAN (cGAN) model for general purpose image-to-image translation, i.e., several problems in image processing like semantic segmentation, sketch-to-image translation, and colorizing images, are all solved by the same network. cGANs involve the conditional generation of images by a generator model. For example, image generation can be conditioned on a class label to generate images specific to that class.

![](https://framerusercontent.com/images/eoiXE8Mt4iJL62rmM9iUXUWkLM.png)

_Source:_ [_Paper_](https://arxiv.org/pdf/1611.07004.pdf)

Pix2pix consists of a [U-Net](https://arxiv.org/pdf/1505.04597.pdf) generator network and a [PatchGAN](https://github.com/He-jerry/PatchGAN) discriminator network, which takes in NxN patches of an image to predict whether it is real or fake, unlike traditional GAN models. The authors argue that such a discriminator enforces more constraints that encourage sharp high-frequency detail. Examples of results obtained by the pix2pix model on image-to-map and map-to-image tasks are shown below.

![](https://framerusercontent.com/images/BurgRfmH13H7EuyyJnsus9rmh0.png)

## **Key Takeaways**

The information technology era we live in has made visual data widely available. However, a lot of processing is required for them to be transferred over the internet or for purposes like information extraction, predictive modeling, etc.

The advancement of deep learning technology gave rise to CNN models, which were specifically designed for processing images. Since then, several advanced models have been developed that cater to specific tasks in the Image Processing niche. We looked at some of the most critical techniques in Image Processing and popular Deep Learning-based methods that address these problems, from image compression and enhancement to image synthesis.

Recent research is focused on reducing the need for ground truth labels for complex tasks like object detection, semantic segmentation, etc., by employing concepts like [Semi-Supervised Learning](https://www.v7labs.com/blog/semi-supervised-learning-guide) and [Self-Supervised Learning](https://www.v7labs.com/blog/self-supervised-learning-guide), which makes models more suitable for broad practical applications.‍


# **CNN (Convolutional Neural Networks)**: 

A ****Convolutional Neural Network (CNN)**** is a type of <font color="#5f497a">Deep Learning neural network architecture</font> commonly used in Computer Vision. Computer vision is a field of Artificial Intelligence that enables a computer to understand and interpret the image or visual data.

When it comes to Machine Learning, Artificial Neural Networks perform really well. Neural Networks are used in various datasets like images, audio, and text. Different types of Neural Networks are used for different purposes, for example for predicting the sequence of words we use ****Recurrent Neural Networks**** more precisely an <font color="#5f497a">LSTM</font>, similarly for image classification we use Convolution Neural networks. In this blog, we are going to build a basic building block for CNN.

### Neural Networks: Layers and Functionality

In a regular Neural Network there are three types of layers:

1. ****Input Layers:**** It’s the layer in which we give input to our model. The number of neurons in this layer is equal to the total number of features in our data (number of pixels in the case of an image).
2. ****Hidden Layer:**** The input from the Input layer is then fed into the hidden layer. There can be many hidden layers depending on our model and data size. Each hidden layer can have different numbers of neurons which are generally greater than the number of features. The output from each layer is computed by matrix multiplication of the output of the previous layer with learnable weights of that layer and then by the addition of learnable biases followed by activation function which makes the network nonlinear.
3. ****Output Layer:**** The output from the hidden layer is then fed into a logistic function like sigmoid or softmax which converts the output of each class into the probability score of each class.

The data is fed into the model and output from each layer is obtained from the above step is called [****feedforward****](https://www.geeksforgeeks.org/understanding-multi-layer-feed-forward-networks/), we then calculate the error using an error function, some common error functions are cross-entropy, square loss error, etc. The error function measures how well the network is performing. After that, we backpropagate into the model by calculating the derivatives. This step is called [****Backpropagation****](https://www.geeksforgeeks.org/backpropagation-in-data-mining/) which basically is used to minimize the loss.

### Convolution Neural Network

Convolutional Neural Network (CNN) is the extended version of artificial neural networks (ANN) which is predominantly used to extract the feature from the grid-like matrix dataset. For example visual datasets like images or videos where data patterns play an extensive role.

### CNN Architecture

Convolutional Neural Network consists of multiple layers like the input layer, Convolutional layer, Pooling layer, and fully connected layers.

![max](https://media.geeksforgeeks.org/wp-content/uploads/20231218174301/max.png)

Simple CNN architecture

The Convolutional layer applies filters to the input image to extract features, the Pooling layer down samples the image to reduce computation, and the fully connected layer makes the final prediction. The network learns the optimal filters through backpropagation and gradient descent.

### How Convolutional Layers Works?

<font color="#5f497a">Convolution Neural Networks</font> or convNet are neural networks that share their parameters. Imagine you have an image. It can be represented as a cuboid having its length, width (dimension of the image), and height (i.e the channel as images generally have red, green, and blue channels). 

![cnn-2-300x133](https://media.geeksforgeeks.org/wp-content/uploads/20231218174321/cnn-2-300x133.jpg)

Now imagine taking a small patch of this image and running a small neural network, called a filter or kernel on it, with say, K outputs and representing them vertically. Now slide that neural network across the whole image, as a result, we will get another image with different widths, heights, and depths. Instead of just R, G, and B channels now we have more channels but lesser width and height. This operation is called ****Convolution****. If the patch size is the same as that of the image it will be a regular neural network. Because of this small patch, we have fewer weights. 

![Screenshot-from-2017-08-15-13-55-59-300x217](https://media.geeksforgeeks.org/wp-content/uploads/20231218174335/Screenshot-from-2017-08-15-13-55-59-300x217.png)

Image source: Deep Learning Udacity

### Mathematical Overview of Convolution

Now let’s talk about a bit of mathematics that is involved in the whole convolution process. 

- Convolution layers consist of a set of learnable filters (or kernels) having small widths and heights and the same depth as that of input volume (3 if the input layer is image input).
- For example, if we have to run convolution on an image with dimensions 34x34x3. The possible size of filters can be axax3, where ‘a’ can be anything like 3, 5, or 7 but smaller as compared to the image dimension.
- During the forward pass, we slide each filter across the whole input volume step by step where each step is called [****stride****](https://www.geeksforgeeks.org/ml-introduction-to-strided-convolutions/) (which can have a value of 2, 3, or even 4 for high-dimensional images) and compute the dot product between the kernel weights and patch from input volume.
- As we slide our filters we’ll get a 2-D output for each filter and we’ll stack them together as a result, we’ll get output volume having a depth equal to the number of filters. The network will learn all the filters.

### ****Layers Used to Build ConvNets****

A complete Convolution Neural Networks architecture is also known as covnets. A covnets is a sequence of layers, and every layer transforms one volume to another through a differentiable function.   
****Types of layers:**** datasets  
Let’s take an example by running a covnets on of image of dimension 32 x 32 x 3. 

- ****Input Layers:**** It’s the layer in which we give input to our model. In CNN, Generally, the input will be an image or a sequence of images. This layer holds the raw input of the image with width 32, height 32, and depth 3.
- ****Convolutional Layers:**** This is the layer, which is used to extract the feature from the input dataset. It applies a set of learnable filters known as the kernels to the input images. The filters/kernels are smaller matrices usually 2×2, 3×3, or 5×5 shape. it slides over the input image data and computes the dot product between kernel weight and the corresponding input image patch. The output of this layer is referred as feature maps. Suppose we use a total of 12 filters for this layer we’ll get an output volume of dimension 32 x 32 x 12.
- [****Activation Layer:****](https://www.geeksforgeeks.org/activation-functions-neural-networks/) By adding an activation function to the output of the preceding layer, activation layers add nonlinearity to the network. it will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are ****RELU****: max(0, x),  ****Tanh****, ****Leaky RELU****, etc. The volume remains unchanged hence output volume will have dimensions 32 x 32 x 12.
- [****Pooling layer:****](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/) This layer is periodically inserted in the covnets and its main function is to reduce the size of volume which makes the computation fast reduces memory and also prevents overfitting. Two common types of pooling layers are ****max pooling**** and ****average pooling****. If we use a max pool with 2 x 2 filters and stride 2, the resultant volume will be of dimension 16x16x12. 

![Screenshot-from-2017-08-15-17-04-02](https://media.geeksforgeeks.org/wp-content/uploads/20231218174414/Screenshot-from-2017-08-15-17-04-02.png)

Image source: cs231n.stanford.edu

- ****Flattening:**** The resulting feature maps are flattened into a one-dimensional vector after the convolution and pooling layers so they can be passed into a completely linked layer for categorization or regression.
- ****Fully Connected Layers:**** It takes the input from the previous layer and computes the final classification or regression task.

![Screenshot-from-2017-08-15-17-22-40](https://media.geeksforgeeks.org/wp-content/uploads/20231218174454/Screenshot-from-2017-08-15-17-22-40.jpg)

Image source: cs231n.stanford.edu

- ****Output Layer:**** The output from the fully connected layers is then fed into a logistic function for classification tasks like sigmoid or softmax which converts the output of each class into the probability score of each class.

## Example: Applying CNN to an Image

Let’s consider an image and apply the convolution layer, activation layer, and pooling layer operation to extract the inside feature.

****Input image:****

![Ganesh](https://media.geeksforgeeks.org/wp-content/uploads/20231218174514/Ganesh.jpg)

Input image

#### Step:

- import the necessary libraries
- set the parameter
- define the kernel
- Load the image and plot it.
- Reformat the image 
- Apply convolution layer operation and plot the output image.
- Apply activation layer operation and plot the output image.
- Apply pooling layer operation and plot the output image.

### Here is a python code for explanation:

<font color="#00b050">import numpy as np</font>
<font color="#00b050">	import tensorflow as tf</font>
<font color="#00b050">	import matplotlib.pyplot as plt</font>
<font color="#00b050">	from itertools import product</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Set up Matplotlib for better visualization</font>
<font color="#00b050">	plt.rc('figure', autolayout=True)</font>
<font color="#00b050">	plt.rc('image', cmap='magma')</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Define the convolution kernel</font>
<font color="#00b050">	kernel = tf.constant([[-1, -1, -1],</font>
<font color="#00b050">	                      [-1,  8, -1],</font>
<font color="#00b050">	                      [-1, -1, -1]])</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Load and preprocess the image</font>
<font color="#00b050">	image = tf.io.read_file('Ganesh.jpg')  # Read the image file</font>
<font color="#00b050">	image = tf.io.decode_jpeg(image, channels=1)  # Decode as a grayscale image</font>
<font color="#00b050">	image = tf.image.resize(image, size=[300, 300])  # Resize the image to 300x300 pixels</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Display the original image</font>
<font color="#00b050">	plt.figure(figsize=(5, 5))</font>
<font color="#00b050">	plt.imshow(tf.squeeze(image).numpy(), cmap='gray')</font>
<font color="#00b050">	plt.axis('off')</font>
<font color="#00b050">	plt.title('Original Gray Scale Image')</font>
<font color="#00b050">	plt.show()</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Prepare the image and kernel for convolution</font>
<font color="#00b050">	image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Convert to float32</font>
<font color="#00b050">	image = tf.expand_dims(image, axis=0)  # Add a batch dimension</font>
<font color="#00b050">	kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])  # Reshape the kernel to 4D</font>
<font color="#00b050">	kernel = tf.cast(kernel, dtype=tf.float32)  # Cast the kernel to float32</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Perform convolution</font>
<font color="#00b050">	conv_fn = tf.nn.conv2d</font>
<font color="#00b050">	image_filter = conv_fn(</font>
<font color="#00b050">	    input=image,</font>
<font color="#00b050">	    filters=kernel,</font>
<font color="#00b050">	    strides=1,  # Convolution stride</font>
<font color="#00b050">	    padding='SAME'  # Padding to preserve image size</font>
<font color="#00b050">	)</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Apply activation (ReLU)</font>
<font color="#00b050">	relu_fn = tf.nn.relu</font>
<font color="#00b050">	image_detect = relu_fn(image_filter)</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Apply max pooling</font>
<font color="#00b050">	pool = tf.nn.pool</font>
<font color="#00b050">	image_condense = pool(input=image_detect,</font>
<font color="#00b050">	                       window_shape=(2, 2),  # Pooling window size</font>
<font color="#00b050">	                       pooling_type='MAX',  # Maximum pooling</font>
<font color="#00b050">	                       strides=(2, 2),  # Pooling stride</font>
<font color="#00b050">	                       padding='SAME')</font>
<font color="#00b050">	</font>
<font color="#00b050">	# Display the results</font>
<font color="#00b050">	plt.figure(figsize=(15, 5))</font>
<font color="#00b050">	plt.subplot(1, 3, 1)</font>
<font color="#00b050">	plt.imshow(tf.squeeze(image_filter))</font>
<font color="#00b050">	plt.axis('off')</font>
<font color="#00b050">	plt.title('Convolution')</font>
<font color="#00b050">	</font>
<font color="#00b050">	plt.subplot(1, 3, 2)</font>
<font color="#00b050">	plt.imshow(tf.squeeze(image_detect))</font>
<font color="#00b050">	plt.axis('off')</font>
<font color="#00b050">	plt.title('Activation')</font>
<font color="#00b050">	</font>
<font color="#00b050">	plt.subplot(1, 3, 3)</font>
<font color="#00b050">	plt.imshow(tf.squeeze(image_condense))</font>
<font color="#00b050">	plt.axis('off')</font>
<font color="#00b050">	plt.title('Pooling')</font>
<font color="#00b050">	plt.show()</font>
<font color="#00b050">	</font>

### Output:

  
![download-(23)](https://media.geeksforgeeks.org/wp-content/uploads/20231218174558/download-(23).png)

Original Grayscale image


  

![Screenshot-from-2023-03-20-15-07-10-(2)](https://media.geeksforgeeks.org/wp-content/uploads/20231218174648/Screenshot-from-2023-03-20-15-07-10-(2).jpg)


### Key Components of a CNN

The convolutional neural network is made of four main parts.

But how do CNNs Learn with those parts?

They help the CNNs mimic how the human brain operates to recognize patterns and features in images:

- Convolutional layers
- Rectified Linear Unit (ReLU for short)
- Pooling layers
- Fully connected layers

This section dives into the definition of each one of these components through the example of the following example of classification of a handwritten digit.

![Architecture of the CNNs applied to digit recognition](https://images.datacamp.com/image/upload/v1700043905/image10_f8b261ebf1.png)


## Advantages and Disadvantages of Convolutional Neural Networks (CNNs)

### Advantages of CNNs:

1. Good at detecting patterns and features in images, videos, and audio signals.
2. Robust to translation, rotation, and scaling invariance.
3. End-to-end training, no need for manual feature extraction.
4. Can handle large amounts of data and achieve high accuracy.

### Disadvantages of CNNs:

1. Computationally expensive to train and require a lot of memory.
2. Can be prone to overfitting if not enough data or proper regularization is used.
3. Requires large amounts of labeled data.
4. Interpretability is limited, it’s hard to understand what the network has learned.

## ==How we can apply CNNs in Drowning Detection?==

**1. Image Feature Extraction:**

- **Input:** Single images captured by cameras or surveillance systems.
- **Process:**
    - Apply a pre-trained CNN model like VGG16 or ResNet to extract features from the image. These features represent specific patterns and shapes within the image.  
        
    - Focus on features that might indicate drowning, such as:
        - **Body posture:** Vertical orientation with arms extended.
        - **Head position:** Not above the water surface.
        - **Movement patterns:** Unusual or lack of movement.

**2. Anomaly Detection:**

- **Input:** Sequence of images or video frames.
- **Process:**
    - Train a CNN model on a dataset of labeled images/frames. Label images as "normal" or "drowning."  
        
    - The model learns to identify features associated with drowning events.  
        
    - Compare extracted features from new frames to the learned patterns.
    - Identify significant deviations as potential drowning incidents.

**3. Advantages of CNNs in Drowning Detection:**

- **Automatic Feature Extraction:** No need for manual feature engineering, saving time and effort.  
    
- **Pattern Recognition:** Efficiently learn complex patterns in images/videos associated with drowning.
- **Robustness:** Can handle variations in lighting, water clarity, and camera angles to some extent.  
    

**4. Challenges and Considerations:**

- **Data Collection:** Requires a large, diverse dataset of labeled images/videos for training.  
    
- **False Positives:** Need to minimize false alarms triggered by splashing, diving, or other activities.
- **Computational Cost:** Training and running CNNs can be computationally expensive on low-power devices.
- **Real-time Processing:** Optimizing models for real-time analysis on surveillance systems.

**5. Additional Techniques:**

- Integrating CNNs with other image processing techniques like object detection and pose estimation can improve accuracy.
- Combining image analysis with sensor data (e.g., water flow sensors) for a more comprehensive approach.


# **Object Detection Algorithms**: 

Object detection algorithms are computer vision techniques that aim to identify and locate objects within an image or video. These algorithms not only classify what is in the image but also determine where the object is located in the image.

## Where is object detection used?

As we go about our daily life, object detection is already all around us. For example, when your smartphone unlocks with face detection. Or in video surveillance of stores or warehouses, it identifies suspicious activities. 

Here are several more major applications of object detection: 

- **Number plate recognition** – using both object detection and optical character recognition (OCR) technology to recognize the alphanumeric characters on a vehicle. You can use object detection to capture images and detect vehicles in a particular image. Once the model detects the number plate, the OCR technology works on converting the two-dimensional data into machine-encoded text.
- **Face detection and recognition** as previously discussed, one of the major applications of object detection is face detection and recognition. With the help of modern algorithms, we can detect human faces in an image or video. It’s now even possible to recognize faces with just a single trained image due to one-shot learning methods.
- **Object tracking** while watching a game of baseball or cricket, the ball could hit far away. In these situations, it’s good to track the motion of the ball along with the distance it’s covering. For this purpose, object tracking can ensure that we have continuous information on the direction of movement of the ball.
- **Self-driving cars** for autonomous cars, it’s crucial to study the different elements around the car while driving. An object detection model trained on multiple classes to recognize the different entities becomes vital for the good performance of autonomous vehicles.
- **Robotics** many tasks like lifting heavy loads, pick and place operations, and other real-time jobs are performed by robots. Object detection is essential for robots to detect things and automate tasks.


#### → Overview of architecture

![HOG object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2024/04/object-detection-algorithms-and-libraries-1.png?resize=1200%2C628&ssl=1)

_HOG – Object Detection Algorithm | [Source](https://www.youtube.com/watch?v=QmYJCxJWdEs)_

Before we understand the overall architecture of HOG, here’s how it works. For a particular pixel in an image, the histogram of the gradient is calculated by considering the vertical and horizontal values to obtain the feature vectors. With the help of the gradient magnitude and the gradient angles, we can get a clear value for the current pixel by exploring the other entities in their horizontal and vertical surroundings.  
  
As shown in the above image representation, we’ll consider an image segment of a particular size. The first step is to find the gradient by dividing the entire computation of the image into gradient representations of 8×8 cells. With the help of the 64 gradient vectors that are achieved, we can split each cell into angular bins and compute the histogram for the particular area. This process reduces the size of 64 vectors to a smaller size of 9 values.  
  
Once we obtain the size of 9 point histogram values (bins) for each cell, we can choose to create overlaps for the blocks of cells. The final steps are to form the feature blocks, normalize the obtained feature vectors, and collect all the features vectors to get an overall HOG feature. Check the following links for more information about this: [[1]](https://www.youtube.com/watch?v=QmYJCxJWdEs) and [[2]](https://www.youtube.com/watch?v=XmO0CSsKg88).

#### → Achievements of HOG

1. Creation of a feature descriptor useful for performing object detection. 
2. Ability to be combined with support vector machines (SVMs) to achieve high-accuracy object detection.
3. Creation of a sliding window effect for the computation of each position.

#### → Points to consider

1. **Limitations** – While the Histogram of Oriented Gradients (HOG) was quite revolutionary in the beginning stages of object detection, there were a lot of issues in this method. It’s quite time-consuming for complex pixel computation in images, and ineffective in certain object detection scenarios with tighter spaces.
2. **When to use HOG?** – HOG should often be used as the first method of object detection to test other algorithms and their respective performance. Regardless, HOG finds significant use in most object detection and facial landmark recognition with decent accuracy.
3. **Example use cases** – One of the popular use cases of HOG is in pedestrian detection due to its smooth edges. Other general applications include object detection of specific objects. For more information, refer to the following [link](https://stackoverflow.com/questions/17159885/histogram-of-oriented-gradients-object-detection).

### 2. Region-based Convolutional Neural Networks (R-CNN)

#### → Introduction

The [region-based convolutional neural networks](https://medium.com/@selfouly/r-cnn-3a9beddfd55a) are an improvement in the object detection procedure from the previous methods of HOG and SIFT. In the R-CNN models, we try to extract the most essential features (usually around 2000 features) by making use of selective features. The process of selecting the most significant extractions can be computed with the help of a selective search algorithm that can achieve these more important regional proposals.

#### → Working process of R-CNN

![RCNN object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/RCNN-object-detection-algorithm.png?ssl=1)

_R-CNN – Object Detection Algorithm | [Source](https://d2l.ai/chapter_computer-vision/rcnn.html)_

The working procedure of the selective search algorithm to select the most important regional proposals is to ensure that you generate multiple sub-segmentations on a particular image and select the candidate entries for your task. The greedy algorithm can then be made use of to combine the effective entries accordingly for a recurring process to combine the smaller segments into suitable larger segments.  
  
Once the selective search algorithm is successfully completed, our next tasks are to extract the features and make the appropriate predictions. We can then make the final candidate proposals, and the convolutional neural networks can be used for creating an n-dimensional (either 2048 or 4096) feature vector as output. With the help of a pre-trained convolutional neural network, we can achieve the task of feature extraction with ease.

The final step of the R-CNN is to make the appropriate predictions for the image and label the respective bounding box accordingly. In order to obtain the best results for each task, the predictions are made by the computation of a classification model for each task, while a regression model is used to correct the bounding box classification for the proposed regions. For further reading and information about this topic, refer to the following [link](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e).

#### → Issues with R-CNN

1. Despite producing effective results for feature extraction with the pre-trained CNN models, the overall procedure of extraction of all the region proposals, and ultimately the best regions with the current algorithms, is extremely slow.  
2. Another major drawback of the R-CNN model is not only the slow rate of training but also the high prediction time. The solution requires the use of large computational resources, increasing the overall feasibility of the process. Hence, the overall architecture can be considered quite expensive.  
3. Sometimes, bad candidate selections can occur at the initial step due to the lack of improvements that can be made in this particular step. A lot of problems in the trained model could be caused by this.

#### → Points to consider

1. **When To Use R-CNN?** – R-CNN similar to the HOG object detection method must be used as a first baseline for testing the performance of the object detection models. The time taken for predictions of images and objects can take a bit longer than anticipated, so usually the more modern versions of R-CNN are preferred. 
2. **Example use cases** – There are several applications of R-CNN for solving different types of tasks related to object detection. For example, tracking objects from a drone-mounted camera, locating text in an image, and enabling object detection in Google Lens. Check out the following [link](https://en.wikipedia.org/wiki/Region_Based_Convolutional_Neural_Networks) for more information.

### 3. Faster R-CNN

#### → Introduction

While the R-CNN model was able to perform the computation of object detection and achieve desirable results, there were some major lackluster elements, especially the speed of the model. So, faster methods for tackling some of these issues had to be introduced to overcome the problems that existed in R-CNN. Firstly, the Fast R-CNN was introduced to combat some of the pre-existing issues of R-CNN.

In the fast R-CNN method, the entire image is passed through the pre-trained Convolutional Neural Network instead of considering all the sub-segments. The region of interest (RoI) pooling is a special method that takes two inputs of the pre-trained model and selective search algorithm to provide a fully connected layer with an output. In this section, we will learn more about the Faster R-CNN network, which is an improvement on the fast R-CNN model.

#### → Understanding Faster R-CNN

![Faster RCNN object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Faster-RCNN-object-detection-algorithm.png?ssl=1)

_Faster R-CNN – Object Detection Algorithm | [Source](https://d2l.ai/chapter_computer-vision/rcnn.html)_

The Faster R-CNN model is one of the best versions of the R-CNN family and improves the speed of performance tremendously from its predecessors. While the R-CNN and Fast R-CNN model make use of a selective search algorithm to compute the region proposals, the Faster R-CNN method replaces this existing method with a superior region proposal network. The region proposal network (RPN) computes images from a wide range and different scales to produce effective outputs.

The regional proposal network reduces the margin computation time, usually 10 ms per image. This network consists of the convolutional layer from which we can obtain the essential feature maps of each pixel. For each feature map, we have multiple anchor boxes which have varying scales, different sizes, and aspect ratios. For each anchor box, we make a prediction of the particular binary class and generate a bounding box for the same.  
  
The following information is then passed through the non-maximum suppression to remove any unnecessary data since many overlaps are produced while creating the feature maps. The output from the non-maximum suppression is passed through the region of interest, and the rest of the process and computation is similar to the working of Fast R-CNN.  

#### → Points to consider

1. **Limitations** – One of the main limitations of the Faster R-CNN method is the amount of time delay in the proposition of different objects. Sometimes, the speed depends on the type of system being used. 
2. **When To Use Faster R-CNN?** – The time for prediction is faster compared to other CNN methods. While R-CNN usually takes around 40-50 seconds for the prediction of objects in an image, the Fast R-CNN takes around 2 seconds, but the Faster R-CNN returns the optimal result in just about 0.2 seconds.
3. **Example use cases** – The examples of use cases for Faster R-CNN are similar to the ones described in the R-CNN methodology. However, with Faster R-CNN, we can perform these tasks optimally and achieve results more effectively.

### 4. Single Shot Detector (SSD)

#### → Introduction

The [single-shot detector](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11) for multi-box predictions is one of the fastest ways to achieve the real-time computation of object detection tasks. While the Faster R-CNN methodologies can achieve high accuracies of prediction, the overall process is quite time-consuming and it requires the real-time task to run at about 7 frames per second, which is far from desirable. 

The single-shot detector (SSD) solves this issue by improving the frames per second to almost five times more than the Faster R-CNN model. It removes the use of the region proposal network and instead makes use of multi-scale features and default boxes. 

#### → Overview of architecture

![SSD object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/SSD-object-detection-algorithm.png?ssl=1)

_SSD – Object Detection Algorithm | [Source](https://arxiv.org/pdf/1512.02325.pdf)_

The single-shot multibox detector architecture can be broken down into mainly three components. The first stage of the single-shot detector is the feature extraction step, where all the crucial feature maps are selected. This architectural region consists of only fully convolutional layers and no other layers. After extracting all the essential feature maps, the next step is the process of detecting heads. This step also consists of fully convolutional neural networks.

However, in the second stage of detection heads, the task is not to find the semantic meaning for the images. Instead, the primary goal is to create the most appropriate bounding maps for all the feature maps. Once we have computed the two essential stages, the final stage is to pass it through the non-maximum suppression layers for reducing the error rate caused by repeated bounding boxes.

#### → Limitations of SSD

1. The SSD, while boosting the performance significantly, suffers from decreasing the resolution of the images to a lower quality. 
2. The SSD architecture will typically perform worse than the Faster R-CNN for small-scale objects.

#### → Points to consider

1. **When To Use SSD?** – The single-shot detector is often the preferred method. The main reason for using the single-shot detector is because we mainly prefer faster predictions on an image for detecting larger objects, where accuracy is not an extremely important concern. However, for more accurate predictions for smaller and precise objects, other methods must be considered.
2. **Example use cases** – The Single-shot detector can be trained and experimented on a multitude of datasets, such as PASCAL VOC, COCO, and ILSVRC datasets. They can perform well on larger object detections like the detection of humans, tables, chairs, and other similar entities. 

### 5. YOLO (You Only Look Once)

#### → Introduction

You only look once ([YOLO](https://pjreddie.com/darknet/yolo/)) is one of the most popular model architectures and algorithms for object detection. Usually, the first concept found on a Google search for algorithms on object detection is the YOLO architecture. There are several versions of YOLO, which we will discuss in the upcoming sections. The YOLO model uses one of the best neural network archetypes to produce high accuracy and overall speed of processing. This speed and accuracy is the main reason for its popularity. 

#### → Working process of YOLO

![YOLO object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/YOLO-object-detection-algorithm.png?ssl=1)

_YOLO – Object Detection Algorithm | [Source](https://arxiv.org/pdf/1506.02640.pdf)_

The YOLO architecture utilizes three primary terminologies to achieve its goal of object detection. Understanding these three techniques is quite significant to know why exactly this model performs so quickly and accurately in comparison to other object detection algorithms. The first concept in the YOLO model is residual blocks. In the first architectural design, they have used 7×7 residual blocks to create grids in the particular image. 

Each of these grids acts as central points and a particular prediction for each of these grids is made accordingly. In the second technique, each of the central points for a particular prediction is considered for the creation of the bounding boxes. While the classification tasks work well for each grid, it’s more complex to segregate the bounding boxes for each of the predictions that are made. The third and final technique is the use of the intersection of union (IOU) to calculate the best bounding boxes for the particular object detection task.

#### → Advantages of YOLO

1. The computation and processing speed of YOLO is quite high, especially in real-time compared to most of the other training methods and object detection algorithms. 
2. Apart from the fast computing speed, the YOLO algorithm also manages to provide an overall high accuracy with the reduction of background errors seen in other methods. 
3. The architecture of YOLO allows the model to learn and develop an understanding of numerous objects more efficiently. 

#### → Limitations of YOLO

1. Failure to detect smaller objects in an image or video because of the lower recall rate.
2. Cant’t detect two objects that are extremely close to each other due to the limitations of bounding boxes. 

#### → Versions of YOLO

The YOLO architecture is one of the most influential and successful object detection algorithms. With the introduction of the YOLO architecture in 2016, their consecutive versions YOLO v2 and YOLO v3 arrived in 2017 and 2018. While there was no new release in 2019, 2020 saw three quick releases: YOLO v4, YOLO v5, and PP-YOLO. Each of the newer versions of YOLO slightly improved on their previous ones. The tiny YOLO was also released to ensure that object detection could be supported on embedded devices.

#### → Points to consider

1. **When To Use YOLO?** – While all the previously discussed methods perform quite well on images and sometimes video analysis for object detection, the YOLO architecture is one of the most preferred methods for performing object detection in real-time. It achieves high accuracy on most real-time processing tasks with a decent speed and frames per second depending on the device that you’re running the program on.
2. **Example use cases** – Some popular use cases of the YOLO architecture apart from object detection on numerous objects include vehicle detection, animal detection, and person detection. For further information, refer to the following [link](https://www.pixelsolutionz.com/application-of-yolo-in-real-life/).

### 6. RetinaNet

#### → Introduction

The [RetinaNet](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4) model introduced in 2017 became one of the best models with single-shot object detection capabilities that could surpass other popular object detection algorithms during this time. When the RetinaNet Architecture was released, the object detection capabilities exceeded that of the Yolo v2 and the SSD models. While maintaining the same speed as these models, it was also able to compete with the R-CNN family in terms of accuracy. Due to these reasons, the RetinaNet model finds a high usage in detecting objects through satellite imagery. 

#### → Overview of architecture

![RetinaNet object detection algorithm](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/RetinaNet-object-detection-algorithm.png?ssl=1)

_RetineNet – Object Detection Algorithm | [Source](https://arxiv.org/pdf/1708.02002.pdf)_

The RetinaNet architecture is built in such a way that the previous issues of single-shot detectors are somewhat balanced out to produce more effective and efficient results. In this model architecture, the cross-entropy loss in the previous models is replaced with the focal loss. The focal loss handles the [class imbalance problems](https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data) that exist in architectures like YOLO and SSD. The RetinaNet model is a combination of three main entities.

RetinaNet is built using three factors, namely the ResNet model (specifically ResNet-101), the feature pyramid network (FPN), and the focal loss. The feature pyramid network is one of the best methods for overcoming a majority of the shortcomings of the previous architecture. It helps in combining the semantic rich features of lower resolution images with that of the semantically weak features of the higher resolution images.

In the final output, we can create both the classification and regression models similar to the other object detection methods discussed previously. The classification network is used for appropriate multi-class predictions, while the regression network is built to predict the appropriate bounding boxes for the classified entities. For further information and reading on this topic, check out the article or video guides respectively from the following links, [[1]](https://developers.arcgis.com/python/guide/how-retinanet-works/) and [[2]](https://www.youtube.com/watch?v=infFuZ0BwFQ).

#### → Points to consider

1. **When to use RetinaNet?** – RetinaNet is currently one of the best methods for object detection in a number of different tasks. It can be used as a replacement for a single-shot detector for a multitude of tasks to achieve quick and accurate results for images.
2. **Example use cases** – There’s a wide array of applications that can be performed with the RetinaNet object detection algorithm. A high-level application of RetinaNet is used for object detection in aerial and satellite imagery. 

## Object detection libraries

### 1. ImageAI

#### → Introduction

The ImageAI library aims to provide developers with a multitude of computer vision algorithms and deep learning methodologies to complete tasks related to object detection and image processing. The primary objective of the ImageAI library is to provide an effective approach to coding object detection projects with a few lines of code.  
  
For further information on this topic, make sure to visit the official documentation of the ImageAI library from the following [link](https://imageai.readthedocs.io/en/latest/#). Most of the code blocks that are available are written with the help of the Python programming language along with the popular deep learning framework Tensorflow. As of June 2021, this library makes use of a PyTorch backend for the computation of image processing tasks.

#### → Overview

The ImageAI library supports a ton of operations related to object detection, namely image recognition, image object detection, video object detection, video detection analysis, Custom Image Recognition Training and Inference, and Custom Objects Detection Training and Inference. The image recognition functionality can recognize up to 1000 different objects in a particular image. 

The image and video object detection task will help to detect 80 of the most common objects seen in daily life. The video detection analysis will help to compute the timely analysis of any particular object that is detected in a video or in real-time. It’s also possible to introduce custom images for training your own samples in this library. You can train a lot more objects for the object detection task with the help of newer images and datasets.

#### → GitHub Reference

For further information and reading on the ImageAI library, refer to the following [GitHub Reference](https://github.com/OlafenwaMoses/ImageAI).

### 2. GluonCV

#### → Introduction

The [GluonCV](https://cv.gluon.ai/) is one of the best library frameworks with most of the state-of-the-art implementations for deep learning algorithms for various computer vision applications. The primary objective of this library is to help the enthusiasts of this field to achieve productive results in a shorter time period. It has some of the best features with a large set of training datasets, implementation techniques, and carefully designed APIs. 

#### → Overview

The GluonCV library framework supports an extensive number of tasks that you can accomplish with it. These projects include image classification tasks, object detection tasks in image, video, or real-time, semantic segmentation and instance segmentation, pose estimation to determine the pose of a particular body, and action recognition to detect the type of human activity being performed. These features make this library one of the best object detection libraries to achieve quicker results.

This framework provides all the state-of-the-art techniques required for performing the previously mentioned tasks. It supports both MXNet and PyTorch and has a wide array of tutorials and additional support from which you can start exploring numerous concepts. It contains a large number of training models from which you can explore and create a particular machine learning model of your choice to perform the specific task.

With either the MXNet or PyTorch installed in your virtual environment, you can follow this [link](https://cv.gluon.ai/install.html) to get you started with the simple installation of this object detection library. You can choose your specific setup for the library. It also allows you access to the Model Zoo, which is one of the best platforms for easy deployment of machine learning models. All these features make GluonCV a great object detection library.

#### → GitHub Reference

For further information and reading on this library, check out the following [GitHub Reference](https://github.com/dmlc/gluon-cv).

### 3. Detectron2

#### → Introduction

The Detectron2 framework developed by Facebook’s AI research (FAIR) team is considered to be a next-generation library that supports most of the state-of-the-art detection techniques, object detection methods, and segmentation algorithms. The [Detectron2](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/) library is a PyTorch-based object detection framework. The library is highly flexible and extensible, providing users with multiple high-quality implementation algorithms and techniques. It also supports numerous applications and production projects on Facebook.

#### → Overview

The Detectron2 library developed on PyTorch by FaceBook has tremendous applications and can be trained on single or multiple GPUs to produce fast and effective results. With the help of this library, you can implement several high-quality object detection algorithms to achieve the best results. These state-of-the-art technologies and object detection algorithms supported by the library include 

DensePose, panoptic feature pyramid networks, and numerous other variations of the pioneering Mask R-CNN model family. [[1]](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)

The Detectron2 library also allows the users to train custom models and datasets with ease. The installation procedure for the following is quite simple. The only dependencies that you require for the following is PyTorch and the COCO API. Once you have the following requirements, you can proceed to install the Detectron2 model and train a multitude of models with ease. For learning more and understanding how exactly you can use the following library, you can use the following [guide](https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578).

### 4. YOLOv3_TensorFlow

#### → Introduction

The YOLO v3 model is one of the successful implementations of the YOLO series, which was released in 2018. The third version of YOLO improves on the previous models. The performance of this model is better than its predecessors in terms of both speed and accuracy. Unlike the other architectures, it can also perform decently on smaller objects with good precision. The only main concern in comparison to other major algorithms is the tradeoff between speed and accuracy.

#### → Overview

The YOLOv3_TensorFlow library is one of the earliest implementations of the YOLO architecture for object detection processing and computing. It provides extremely fast GPU computations, effective results and data pipelines, weight conversions, faster training times, and a lot more. While the library can be obtained from the link provided in the next section, the support has stopped for this framework (similarly to most others) and it’s now supported with PyTorch instead.

### 5. Darkflow

#### → Introduction

Darkflow is inspired by the darknet framework and is basically a translation to suit the Python programming language and TensorFlow for making it accessible to a wider range of audiences. Darknet is an early implementation of an object detection library with C and CUDA. The installation and working procedures for this library are quite simple and easy to perform. The framework also supports both CPU and GPU computations of object detection tasks to achieve the best results in either scenario.

#### → Overview

The dark flow framework requires some basic necessities for its implementation. Some of these basic requirements are Python3, TensorFlow, Numpy, and Opencv. With these dependencies, you can start computing tasks related to object detection with ease. With the dark flow library, you can achieve a lot of tasks. The dark flow framework has access to YOLO models, and you can download custom weights for a variety of models.

Some of the tasks that the darkflow library helps you accomplish include parsing the annotations, designing the network according to a specific configuration, plotting the graphs with flow, training a new model, training on a custom dataset, creating a real-time or video file, using the Darkflow framework for other similar applications, and finally, it also allows you to save these models in the protobuf (.pb) format.


## ==Can we combine Object Detection and CNNs for Drowning Detection?==

***Integrating object detection algorithms with Convolutional Neural Networks (CNNs) can significantly enhance the accuracy and efficiency of drowning detection systems.***

### 1. **Object Detection for Initial Screening:**

- **Identify Potential Drowning Victims:** Use object detection algorithms like YOLOv8 or Faster R-CNN to locate human figures within video frames.
- **Filter Out False Positives:** Focus on individuals near water bodies or in potential danger zones.
- **Crop Relevant Regions:** Extract regions of interest (ROIs) around detected human figures for further analysis.

### 2. **CNN for Drowning Posture Analysis:**

- **Feature Extraction:** Use CNNs to extract relevant features from the cropped images, such as:
    - **Body Orientation:** Analyze the angle of the body relative to the water surface.
    - **Arm and Leg Position:** Detect unusual limb positions, like outstretched arms or legs.
    - **Head Position:** Identify if the head is submerged or tilted at an unusual angle.
- **Classification:** Train the CNN to classify the extracted features as "drowning" or "non-drowning."

### 3. **Real-time Implementation:**

- **Video Stream Processing:** Process video frames in real-time to identify potential drowning incidents.
- **Efficient Inference:** Use optimized CNN models and hardware acceleration (e.g., GPUs, TPUs) to ensure low latency.
- **Alert System:** Trigger alarms or notifications when a potential drowning event is detected.

### Additional Considerations:

- **Data Collection:** Collect a diverse dataset of videos capturing various water activities, including both normal and drowning scenarios.
- **Data Annotation:** Accurately label the dataset to train the CNN model effectively.
- **Model Training:** Train the CNN model on the labeled dataset using techniques like transfer learning to leverage pre-trained models.
- **Model Deployment:** Deploy the trained model on edge devices or cloud-based platforms for real-time monitoring.

### Example Architecture:

1. **Video Input:** Capture video footage from surveillance cameras or drones.
2. **Object Detection:** Use YOLOv8 to detect human figures in each frame.
3. **ROI Extraction:** Crop the regions of interest around detected humans.
4. **CNN Analysis:** Feed the cropped images into a pre-trained CNN (e.g., ResNet) to extract features.
5. **Classification:** Use a classifier (e.g., SVM, Random Forest) or a dedicated CNN layer to classify the extracted features as "drowning" or "non-drowning."
6. **Alert Generation:** If the classification result indicates a high probability of drowning, trigger an alarm or notification


# **Optical Flow Analysis:**

***Optical flow estimation is used in computer vision to<font color="#7030a0"> characterize and quantify</font> the motion of objects in a video stream, often for motion-based object detection and tracking systems. Moving object detection in a series of frames using optical flow.***

### How Optical Flow Works:

1. **Definition**: Optical flow represents the distribution of apparent velocities of objects in a scene caused by relative motion between the observer and the scene.
    
2. **Assumptions**: Optical flow assumes that the intensity of the pixel remains constant between consecutive frames and that nearby pixels have similar motion.
    
3. **Calculation**: Optical flow algorithms calculate the displacement of pixels between frames by solving an optical flow equation that relates the image gradients to the velocities.
    

### Applications of Optical Flow Analysis:

1. **Object Tracking**: Optical flow can be used to track objects by estimating their motion in consecutive frames, enabling the prediction of their future positions.
    
2. **Motion Estimation**: It helps estimate the motion of objects in a scene, which is useful for video stabilization, object detection, and action recognition.
    
3. **Depth Estimation**: By analyzing the motion of objects in a scene, optical flow can be used to estimate the depth of objects in a 3D space.
    
4. **Video Compression**: Optical flow is used in video compression techniques to predict and encode motion between frames efficiently.
    

### Optical Flow Algorithms:

1. **Lucas-Kanade Method**: A classic and widely used optical flow algorithm that estimates local image motion by assuming that the flow is essentially constant in a local neighborhood of the pixel.
    
2. **Horn-Schunck Method**: An algorithm that computes a global smooth flow field by minimizing the difference between the observed image intensities and the predicted intensities.
    
3. **Farneback Method**: A dense optical flow algorithm that provides dense flow vectors for each pixel in the image.
    

### Lets see some Techniques:

# Compute Optical Flow Velocities

This example shows how to compute the optical flow velocities for a moving object in a video or image sequence.
Read two image frames from an image sequence into the MATLAB workspace.

<font color="#00b050">I1 = imread('car_frame1.png');</font>
<font color="#00b050">	I2 = imread('car_frame2.png');</font>
<font color="#00b050">	</font>
<font color="#00b050">	modelname = 'ex_blkopticalflow.slx';</font>
<font color="#00b050">	open_system(modelname)</font>

![](https://www.mathworks.com/help/examples/vision/win64/ComputeOpticalFlowVelocitiesExample_01.png)

The model reads the images by using the Image From Workspace block. To compute the optical flow velocities, you must first convert the input color images to intensity images by using the Color Space Conversion block. Then, find the velocities by using the `Optical Flow` block with these parameter values:

- **Method** - `Horn-Schunck`
    
- **Compute optical flow between** - `Two images`
    
- **Smoothness factor** - `1`
    
- **Stop iterative solution** - `When maximum number of iterations is reached`
    
- **Maximum number of iterations** - `10`
    
- **Velocity output** - `Horizontal and vertical components in complex form`
    

Overlay both the image frames by using the Compositing block and use the overlaid image to plot the results.

<font color="#00b050"> out = sim(modelname);</font>

<font color="#00b050">Vx = real(out.simout);</font>
<font color="#00b050">	Vy = imag(out.simout);</font>
<font color="#00b050">	img = out.simout1;</font>

<font color="#00b050">flow = opticalFlow(Vx,Vy);</font>

<font color="#00b050">figure</font>
<font color="#00b050">imshow(img)</font>
<font color="#00b050">hold on</font>
<font color="#00b050">plot(flow,'DecimationFactor',[5 5],'ScaleFactor',40)</font>

![](https://www.mathworks.com/help/examples/vision/win64/ComputeOpticalFlowVelocitiesExample_02.png)

### Challenges:

1. **Aperture Problem**: Occurs when the optical flow information is ambiguous due to the projection of 3D motion onto the 2D image plane.
    
2. **Motion Discontinuities**: Optical flow algorithms may struggle with abrupt motion changes or occlusions in the scene.

# ==How we can integrate CNNs , Objection Detection Algorithms & Optical Flow to manipulate Drowning Detection Model?==

### 1. **Object Detection for Initial Screening:**

- **Identify Potential Victims:** Utilize object detection algorithms like YOLOv8 or Faster R-CNN to locate human figures within video frames.
- **Focus on Regions of Interest:** Crop the regions of interest (ROIs) around detected individuals for further analysis.

### 2. **CNN for Drowning Posture Analysis:**

- **Feature Extraction:** Employ a pre-trained CNN (e.g., ResNet, VGG16) to extract relevant features from the cropped images.
- **Drowning Posture Classification:** Train a CNN classifier to distinguish between normal swimming and drowning postures, focusing on:
    - Body orientation: Vertical or horizontal
    - Arm and leg movements: Unusual or erratic motions
    - Head position: Submerged or above water

### 3. **Optical Flow Analysis for Motion Patterns:**

- **Track Object Movement:** Use optical flow techniques to track the movement of detected individuals.
- **Identify Abnormal Motion:** Analyze the speed, direction, and acceleration of the object's motion.
- **Detect Unusual Patterns:** Flag sudden stops, rapid changes in direction, or unusual trajectories.

### 4. **Fusion of Information:**

- **Combine Multiple Cues:** Integrate the outputs from object detection, CNN-based classification, and optical flow analysis.
- **Prioritize Alerts:** Assign higher priority to alerts triggered by multiple cues, such as a person in a drowning posture and exhibiting abnormal motion.
- **Reduce False Positives:** Implement mechanisms to filter out false alarms, such as considering environmental factors like water conditions and weather.

### 5. **Real-Time Implementation:**

- **Efficient Processing:** Utilize optimized CNN architectures and hardware acceleration techniques (e.g., GPUs, TPUs) for real-time processing.
- **Edge Computing:** Deploy the system on edge devices (e.g., cameras, drones) to reduce latency and enable real-time decision-making.
- **Cloud-Based Processing:** Leverage cloud computing resources for more complex analysis and large-scale deployment.

### Example Pipeline:

1. **Video Input:** Capture video footage from surveillance cameras or drones.
2. **Object Detection:** Use YOLOv8 to detect human figures.
3. **ROI Extraction:** Crop the regions of interest around detected individuals.
4. **CNN Analysis:** Feed the cropped images into a pre-trained CNN to classify the posture.
5. **Optical Flow Analysis:** Apply optical flow techniques to track the movement of detected individuals.
6. **Fusion:** Combine the results from object detection, CNN classification, and optical flow analysis to make a final decision.
7. **Alert Generation:** Trigger an alarm if a drowning event is detected.

By integrating these techniques, we can create a robust and accurate drowning detection system that can significantly improve water safety.




"Thank you for reading!"
"Hope you enjoyed this article!"
"Created by Ahmed Waleed"