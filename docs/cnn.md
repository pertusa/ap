# Introduction to Convolutional Neural Networks (CNN)

Convolutional Neural Networks (CNNs) are specifically tailored for computer vision tasks (classification, detection, segmentation, synthesis, etc.) (See Chapter 10.1 [[FDL2023]](https://www.bishopbook.com)). In 1989, LeCun proposed LeNet, a CNN for recognizing handwritten digits in images that was trained by backpropagation. It was widely recognised as the first CNN model achievieng outstanding results matching the performance of support vector machines, then a dominant approach in supervised learning. It laid the foundation for modern CNN architectures and demonstrated the power of convolutional layers and their ability to learn spatial hierarchies of features in an image, a principle that remains central in modern CNNs used for more complex tasks in computer vision.  

However, CNNs got popular in 2012 when outperformed other models at ImageNet Challenge Competition in object classification/detection (here you can see a visualization hierarchy of [1000 classes](https://observablehq.com/@mbostock/imagenet-hierarchy) from ImageNet). Specifically, the first CNN to achieve a breakthrough in the ImageNet Challenge was AlexNet (designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton). It significantly outperformed the other competitors in the 2012 challenge, reducing the error rate by a large margin compared to traditional image classification methods (see here the [results](https://image-net.org/challenges/LSVRC/2012/results.html). 

Today, the [results](https://paperswithcode.com/sota/image-classification-on-imagenet) of the DNN (including CNNs and others as Visual Transformers) in ImageNet Challege Competion are achieving amazing results (more than 95% Top-1 accuracy).  

## Objectives

* O1. Know and implement the fundamental components that conform a CNN.
* O2. Learn about modern convolutional networks that have set milestones in design aspects and how to train them


## 1.  First session of this block (7th February 2024)

### Contents to prepare before (online)

The contents of this first session are related to the objetive 1, being the following:

#### 1.1 Introduction
	[This part can take about 1 hour 🕒️ of personal working.]

* Convolutions 
	* Convolution (or cross-correlation operation): [[DDL23, Section 7.1.3]](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#convolutions)
	* Examples of kernels: [image kernels](https://setosa.io/ev/image-kernels/)

	

* Why CNN? [[UDL2023, Section 10 at beginning]](https://udlbook.github.io/udlbook/),[[DDL23, Section 7.1]](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#from-fully-connected-layers-to-convolutions), [[DDL23, Section 7.1.2]](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#constraining-the-mlp).
	* Reduction of learning parameters 
	*  Invariance: [[UDL2023, Section 10.1]](https://udlbook.github.io/udlbook/), [[DDL23, Section 7.1.2.1]](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#invariance)
	* Locality principle: [[DDL23, Section 7.1.2.2]](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#locality)


	**_NOTE_:** 
	**You don't need to go deeper into the mathematical formulation**

* Architecture of a CNN: a typical CNN has 4 layers: Input layer, Convolution layer, Pooling layer and Fully connected layer. 


#### 1.2 Convolutional layer

	[This part can take about 2 hours 🕒️ of personal working.]

A convolutional layer is the fundamental building block of a CNN. It is able to detect features such as edges, textures, or more complex patterns in higher layers from the input images, extracting characteristics from the previous layers (input layer, previous convolutions,…). A very interesting description of these layers could be found in this [web](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1). It includes the following concepts:

* Padding: Image (n,m), filter (f,f), padding p -> Out (n+2*p-f+1, m+2*p-f+1)
* Strides: Image (n,m), filter (f,f), padding p, stride s -> floor((n+2*p-f)/s+1), floor((m+2*p-f)/s+1)
* Convolutions over volumes
* Multiple filters

After the convolution operation, the non-linearity is introduced by an **activation function** (Sigmoid, ReLU, etc.). It allows to learn more complex patterns.

Some of the contents are extracted from this paper. A summarized version of the concepts can be found in the section 10.2 (except 10.2.6 and 10.2.8) from [[FDL2023]](https://www.bishopbook.com). Finally, to go in depth with these concepts, I recomnend you to read the section [Convolutions for images](
https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#convolutions-for-images), [Padding and Stride](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html#padding-and-stride) and [Multiple Input and Multiple Output](https://d2l.ai/chapter_convolutional-neural-networks/channels.html#multiple-input-and-multiple-output-channels) from the Chapter 7 of the [[DDL2023]](https://D2L.ai) book.

**Notes:** 

* The result of a convolution filter with size $fxf$ to an image of $(h,w)$ size with a padding $p$ is:
	$ \left((h+2p-f)+1, (w+2p-f)+1\right) $
* The result of a convolution filter with size $fxf$ to an image of $(h,w)$ size with a padding $p$ and a stride of $s$ is: $ (\lfloor \div{h+2p-f}{s}+1 \rfloor, lfloor \div{w+2p-f}{s}+1 \rfloor $
* Let $c_{l-1}$ the number of channels of the previous layer $l$ of a convolutonal layer, $f$ the filter height and widht and $c$ the number of filters in the layer. The number of parameters of the convulational layer is: $(f x f x c_{l-1} + 1) x c_l$

##### Exercise

1. Calculate the size of the filters of the different layers (a,b,c), (d,e,f) and (g,h,i) of the following image:

<figure markdown>
  ![Image title]("../images/exercise1.svg"){ width="300" }
  <figcaption>Image caption</figcaption>
</figure>


#### 1.3 Pooling layer

	[This part can take about 1 hour 🕒️ of personal working.]

The main function of the pooling layer is to reduce the spatial dimensions (i.e., width and height) of the input volume for the next convolutional layer. This reduction is achieved without affecting the number of filters in the layer. The pooling operation provides several benefits:

1. **Reduction of computation**: By reducing the dimensions of the feature maps, pooling layers decrease the number of parameters and computations in the network, leading to improved computational efficiency.
1. **Reduction of Overfitting**: Smaller input sizes mean fewer parameters, which can help reduce the overfitting in the network.
1. **Invariance to Transformations**: Pooling helps the network to become invariant to small transformations, distortions, and translations in the input image. This means that the network can recognize the object even if it's slightly modified in different input images.

There are several types of pooling, but the most common are:

- Max Pooling
- Average Pooling

A more detailed explanation could be found in the [Pooling section](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html#pooling) from the Chapter 7 of the [[DDL2023]](https://D2L.ai) book.

**Notes:** 

* There is not parameters to learn!


### Contents for the presential class
In the laboratory class (2 hours 🕒️ duration), we will see how certain components of a convolutional neural network are implemented <a href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab\"></a>.

The aim is for the notebooks to be studied and modified. A later class will present a more advanced practice that will involve modifying and implementing CNN code.

## 2.  Second session of this block (14th February 2024)

### Contents to prepare before (online)

The contents of this first session are related to the objetive 2, being the following:

#### 2.1 Introduction

	[This part can take about 1 hour 🕒️ of personal working.]

A typical CNN has several convolution plus pooling layers, each responsible for feature extraction at different levels of abstraction: filters in first layer detect horizontal, vertical, and diagonal edge; filters in the next layer detect shapes; filters in the following layers detect collection of shapes, etc.

A good starting point to understand the architecture of a simple CNN is to study the [LeNet model](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html#convolutional-neural-networks-lenet). 





##### Exercise

1. Calculate the number of the learning parameters of LeNet architecture


## Biblography

### Textbooks

1. [[DDL2023]](https://D2L.ai) Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J. Dive into Deep Learning. Cambridge University Press (2023)
2. [[UDL2023]](https://udlbook.github.io/udlbook/) Simon J.D. Prince. Understandig Deep Learning. MIT Press (2023).
3. [[FDL2023]](https://www.bishopbook.com) Bishop, C.M., Bishop, H. (2024). Convolutional Networks. In: Deep Learning. Springer, Cham. https://doi.org/10.1007/978-3-031-45468-4_10 (2023)

### Webpages

* https://cs231n.github.io/convolutional-networks/


### Others
* https://arxiv.org/pdf/1603.07285.pdf: Extra information for Convolutional parameters



