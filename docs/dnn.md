**FROM SHALLOW TO DEEP NEURAL NETWORKS**

**Objectives**

- To understand the fundamentals of a basic multilayer perceptron neural network **MLP/ Shallow neural network**. 
- To understand the concept of a “**basic neuron**” and the concepts of “**activation function**”, “**loss function**” and by extension the “**cost function**”. 
- To understand the concept of “**gradient descent”** through the use of function **derivatives** and **the chain rule**. 
- To understand the idea of **forward and backward propagation** as a basis for the search for minima in the gradient descent process. 
- To analyze the basis of **deep neural networks** with multiple inner layers.
- Understanding hyperparameters in a deep neural network and some techniques for improving error.

**Contents**

- Supervised learning fundamentals (cross validation, overfitting, …)
- Logistic Regression fundamentals
- Activation Function types
- Loss function - Cost function
- Gradient Descent
- Logistic Regression Derivatives (chain rule)
- Forward and Backward propagation 
- MLP (Multilayer Perceptron)/ Shallow neural networks
- Deep L-layer Neural Networks
- Dropout, early stopping, parameter initialization

**Methodology**

- **Self-study** **of the proposed** **theoretical materials**. These materials are structured using bibliographical references to explain the theoretical concepts.
- **Moodle Tests**: Moodle tests are proposed to evaluate the theoretical **contents** and to guarantee the learning to tackle the practical part. These moodle tests **must be taken before the practical session**.
- **Workbooks**: Several practical workbooks are proposed to improve the skills of design, implementation and configuration of neural network models.



**Recommended reading for self-study of theoretical content**

To understand the proposed theoretical contents, the following bibliographic resources are proposed for reading: (Three and a half hours)

- **For a quick introduction** to the concept of neural networks, activation functions, loss function, gradient descent, backpropagation, chain rule, hyperparameters, we recommend reading: (30 minutes)
  - [Brief Introduction to Artificial Neural Networks. Section 3.](https://eduscol.education.fr/sti/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14500/14500-brief-introduction-to-artificial-neural-networks-ensps.pdf)
- To **develop** the **concepts** of **linear regression**, loss function, gradient descent and linear regression understood as a basic neural network, the following reading is proposed: (30 minutes)
  - Dive into Deep Learning. Section 3.1. Linear Regression.
  - <https://d2l.ai/chapter_linear-regression/linear-regression.html>
- To **learn** the **concept** of training and **generalization** error, **overfitting** or Cross-Validation we recommend reading: (30 minutes)
  - Dive into Deep Learning. Section 3.6. Generalization.
  - <https://d2l.ai/chapter_linear-regression/generalization.html>
- To **introduce** the applications of neural networks in **classification problems** and the concepts of "Softmax Regression for Classification", "Loss function for classification" and "Cross-Entropy" we recommend reading: (60 minutes)
  - Dive into Deep Learning. Section [4. Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/index.html) 
  - <https://d2l.ai/chapter_linear-classification/index.html>
- To **introduce** the fundamental of **multilayer perceptron** the following reading is proposed: (60 minutes)
  - Dive into Deep Learning. Section [5.](https://d2l.ai/chapter_linear-classification/index.html) Multilayer Perceptrons
  - <https://d2l.ai/chapter_multilayer-perceptrons/index.html>
  - 5.1-5.2: Multilayer Perceptrons. Incorporating Hidden Layers, activation Functions (ReLU, Sigmoid; Tanh).
  - 5.3: Forward Propagation, Backward Propagation, and Computational Graphs
  - 5.4: Numerical Stability and Initialization 
  - 5.5-5.6: Generalization and dropout

**References**

- **Dive into Deep Learning**. Interactive deep learning book with code, math, and discussions Implemented with PyTorch, NumPy/MXNet, JAX, and TensorFlow <https://d2l.ai/> 
- [Brief Introduction to Artificial Neural Networks. Culture Sciences del’ Ingénieur.](https://eduscol.education.fr/sti/sites/eduscol.education.fr.sti/files/ressources/pedagogiques/14500/14500-brief-introduction-to-artificial-neural-networks-ensps.pdf)


**Moodle Test**

- The moodle test must be taken **in advance of the practice session** on February 5. 
- The test will be open between **February 1st at 9:00 am and February 4th at 11:59 pm**. 
- The test has a **maximum duration of 75 minutes** from the start.
- The test consists of **24 triple choice questions**. 
- **Each wrong answer subtracts 1/3** of the value of a correct answer.
- **The mark for the test will be considered as one of the marks for the theoretical part** of the course. See the overall evaluation of the course in the general conditions.

- Before you start the moodle test **make sure you know all these concepts**.

  - About Machine Learning Problems (Supervised; Unsupervised; Regression; Classification)
  - About Training/test datasets and training/generalization error.
  - About Cross-Validation
  - About Over-fitting/under-fitting
  - About the mathematical model for a neuron of a perceptron 
  - About bias 
  - About activation functions 
  - About loss and cost functions expressions
  - About gradient descent
  - About learning rate
  - About forward and back-propagation
  - About derivatives for back-propagation
  - About Hyperparameters 
  - About Dropout
  - About parameter initialization
  - About early stopping

- **To start the Moodle test** go to "UACloud" -> "Moodle" -> "Aprendizaje Profundo" -> Test - FROM SHALLOW TO DEEP Neural Networks


**Workbooks**

For the practical part of the course, **two notebooks** have been prepared to exercise the theoretical concepts acquired. The objective of the notebooks is to allow students to improve their skills in the implementation of neural network models from scratch. Two notebooks are proposed.

- **Notebook 1**: (60 minutes) (30 minutes in class + 30 minutes at home)
  - What is a Perceptron?
  - The main difference between activation functions.
  - The main difference between loss functions
  - Gradient Descent and Learning Rate
  - We will work with a single-layer network with d input nodes and a single output node
  - Shallow Networks / Multilayer Perceptron (MLP)
  - (https://colab.research.google.com/drive/122IssQ-f4AerVSiQHq8BnQcqouQB2T-D?usp=sharing)

- **Notebook 2**: (90 minutes) (60 minutes in class + 30 minutes at home)
  - In this practice we will work with deep neural networks, i.e. with more than two hidden layers. 
  - In addition, we will continue to introduce and put into practice fundamental concepts of deep learning.
  - Overfitting / Underfitting
  - Parameter Initialization
  - Early Stopping
  - Dropout
  - (https://colab.research.google.com/drive/1cy8uvdx8aIIj86Z3X11HmVWogTUqLe3P?usp=sharing)

- To work with the practice you will need to make a local copy of the notebook and answer all the questions that are posed. 
- For the delivery you will have to send a copy of the notebook with the answers. You will have to use the delivery control of "UACloud -> Evaluación" called "FROM SHALLOW TO DEEP NN. NoteBook X".
- The **deadline** for delivery of **notebooks 1 and 2** will be on Wednesday, February 12.
