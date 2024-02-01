# Few and Zero Shot Learning

!!! danger

    This is work in progress. The contents of this page are not final and, therefore, it is not recommended to start working on its contents yet.

The following contents are copied (and slighly adapted) to the AI master from the [ISMIR 2022 tutorial](https://github.com/music-fsl-zsl/tutorial) created by Yu Wang, Hugo Flores García, and Jeong Choi. This is shared under [Creative Commons BY-NC-SA 4.0](https://github.com/music-fsl-zsl/tutorial/blob/main/LICENSE).

## What is Few-Shot Learning (FSL) and Zero-Shot Learning (ZSL)?

Before we dive right into FSL and ZSL, we would like to start with a brief discussion about the labeled data scarcity problem to illustrate the motivation and relevance of few-shot and zero-shot learning.  

Deep learning has been highly successful in data-intensive applications, but is often hampered when the dataset is small. A deep model that generalizes well typically needs to be trained on a large amount of labeled data. However, most MIR datasets are small in size compared to datasets in other domains, such as image and text. This is not only because collecting musical data may be riddled with copyright issues, but annotating musical data can also be very costly. The annotation process often requires expert knowledge and takes a long time as we need to listen to audio recordings multiple times. Therefore, many current MIR studies are built upon relatively small datasets with less-than-ideal model generalizability. MIR researchers have been studying strategies to tackle this scarcity issue for labeled data. These strategies can be roughly summarized into two categories:

- **Data**: crowdsourcing {cite}`levy2011`{cite}`cartwright2015`, data augmentation {cite}`mcfee2015`{cite}`Uhlich2017`, data synthesis {cite}`cartwright2018`{cite}`manilow2019cutting`, etc.
- **Learning Paradigm**: transfer learning {cite}`pons2018`, semi-supervised learning {cite}`li2004`{cite}`You2007`, etc.

However, there are different challenges for each of these approaches. For example, crowdsourcing still requires a large amount of human effort with potential label noise, the diversity gain from data augmentation is limited, and models trained on synthetic data might have issues generalizing to real-world audio.
Even with the help of transfer learning or unsupervised learning, we often still need a significant amount of labeled data (e.g. hundreds of thousands of examples) for the target downstream tasks, which could still be hard for rare classes. 

Few-shot learning (FSL) and zero-shot learning (ZSL), on the other hand, tackle the labeled data scarcity issue from a different angle. They are learning paradigms that aim to learn a model that can learn a new concept (e.g. recognize a new class) quickly, based on just *a handful of labeled examples* (few-shot) or some *side information or metadata* (zero-shot). 

### An example
To have a better idea of how FSL and ZSL differ from standard supervised learning, let's consider an example of training a musical instrument classifier. We assume that there is an existing dataset in which we have abundant labeled examples for common instruments.

- In a standard supervised learning scenario, we train the classification model on the training set `(guitar, piano)`, then at test time, the model classifies *"new examples"* of *"seen classes"* `(guitar, piano)`.
- In a few-shot learning scenario, we also train the model with the same training set that is available `(guitar, piano)`. But at test time, the goal is for the few-shot model to recognize *"new examples"* from *"unseen classes"* (like `(banjo, kazoo)`) by providing a very small amount of labeled examples.  
- In a zero-shot learning scenario, we train the model on the available training set `(guitar, piano)`. But at test time, the goal is for the zero-shot model to recognize *"new examples"* from *"unseen classes"* (like `(banjo, kazoo)`) by providing some side information or metadata (e.g. the instrument family, e.g. string, wind, percussion, etc.) or a text embedding.

![Alt text](images/fsl/FZSL_tutorial_fig.png)


## Few-Shot Learning Foundations

It should come with no surprise that data is at the core of few-shot learning problems. This chapter covers the foundations of few-shot learning – namely how we think about and structure our data – when trying to learn novel, unseen classes with very little labeled data.

When solving traditional classification problems, we typically consider a closed set of classes. That is, we expect to see the same set of classes during inference as we did during training. Few-shot learning breaks that assumption and instead expects that the classification model will encounter novel classes during inference. There is one caveat: there are a **few labeled examples** for each novel class at inference time. 

![thinking-about-data](images/fsl/foundations/thinking-about-data.png)


In few-shot learning, we expect to see **novel** classes at inference time. We also expect to see a few labeled examples (a.k.a. "shots") for each of the novel classes. 

> Transfer learning and data augmentation are often considered approaches to few-shot learning  {cite}`song2022comprehensive`, since both of these approaches are used to learn new tasks with limited data. 
However, we believe these approaches are extensive and deserve their own treatment, and so we will not cover them here.
Instead, we will focus on the topic of **meta-learning** – or learning to learn – which is at the heart of recent advances for few-shot learning in MIR {cite}`wang2020fewshotdrum,flores2021leveraging,wang2022fewshot`. Transfer learning and data augmentation are orthogonal to meta-learning and can be used in conjunction with meta-learning approaches.


### Defining the Problem

Consider that we would like to classify between \(K\) classes, and we have exactly \(N\) labeled examples for each of those classes. 
We say that few-shot models are trained to solve a \(K\)-way, \(N\)-Shot classification task. 

![Support query](images/fsl/foundations/support-query.png)

A few-shot learning problem splits data into two separate sets: the support set (the few labeled examples of novel data) and the query set (the data we want to label).

Few shot learning tasks divide the few labeled data we have and the many unlabeled data we would like to to label into two separate subsets: the **support set** and the **query set**. 

The small set of labeled examples we are given at inference time is the **support set**. The support set is small in size and contains a few (\(N\)) examples for each of the classes we would like to consider. The purpose of the support set is to provide some form of guidance to help the model learn and adapt to the novel classification task. 

Formally, we define the support set as a set of labeled training pairs \(S = \{(x_1, y_1,), (x_2, y_2), ..., (x_N, y_N)\}\), where:

- \(x_i \in \mathbb{R}^D\) is a \(D\)-dimensional input vector.
- \(y_i \in \{1,...,C\}\) is the class label that corresponds to \(x_i\).
- \(S_k\) refers to the set of examples with label \(K\).
- \(N\) is the size of the support set, where \(N = C \times K\).  

On the other hand, the query set – typically denoted as \(Q\) – contains all of the examples we would like to label. We can compare the model's predictions on the query set to the true labels (i.e., ground truth) to compute the loss used for training the model. In evaluation, we can use these predictions to compute metrics such as accuracy, precision, and recall.


#### The Goal

![The goal](images/fsl/foundations/fsl-the-goal.png)

The goal of few-shot learning algorithms is to learn a classification model \(f_\theta\) that is able to generalize to a set of \(K\) previously unseen classes at inference time, with a small support set of \(N\) examples for each previously unseen class.

### Meta Learning - Learning to Learn

In order for a classifier to be able to learn a novel class with only a few labeled examples, we can employ a technique known as **meta learning**, or learning to learn.

> Even though our goal in few shot learning is to be able to learn novel classes with very few labeled examples, we *still* require a sizable training dataset with thousands of examples. The idea is that we can *learn how to learn new classes* from this large training set, and then apply that knowledge to learn novel classes with very few labeled examples.

#### Class-conditional splits

![Class-conditional splits](images/fsl/foundations/class-conditional-splits.png)

In supervised learning, one typically creates a train/test split in the dataset while ensuring that the classes seen during training are the same as those seen during testing.

In few-shot learning, because we'd like our model to generalize to novel classes at inference time, we must make sure that there is no overlap between classes in our train, and test sets.

A train/test split with no overlap between classes is called a **class-conditional** split. 


#### Episodic Training

To take full advantage of a large training set for few-shot learning, we use a technique referred to as **episodic training** {cite}`vinyals2016matching, ravi2017optimization`. 

![Episodic training](images/fsl/foundations/episodic-training.png)

> Episodic training is an efficient way of leveraging a large training dataset to train a few-shot learning model.

Episodic training aims to split each training iteration into it's own self-contained learning task. An episode is like a simulation of a few-shot learning scenario, typically with \(K\) classes and \(N\) labeled examples for each class -- similar to what we expect the model to be able to infer at inference time. 

During episodic training, our model will see a completely new \(N\)-shot, \(K\)-way classification task at each step. To build a single episode, we must sample a completely new support set and query set during each training step. 

Practically, this means that for each episode, we have to choose a subset of \(K\) classes from our training dataset and then sample \(N\) labeled examples (for the support set) and \(q\) examples (for the query set) for each class that we randomly sampled. 


### Evaluating a Few-Shot Model
Validation and Evaluation during episodic training can be done in a similar fashion to training. We can build a series of episodes from our validation and evaluation datasets, and then evaluate the model on each episode using standard classifcation metrics, like [precision, accuracy, F-score,](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) and [AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc). 

We've now covered the basic foundations of few-shot learning. In the next chapter, we'll look at some of the most common approaches to few-shot learning, namely **metric**-based, **optimization**-based, and **memory**-based approaches. 

-------------------------

# Approaches

Now that we have a grasp of the foundations of few-shot learning,  we'll take a look at some of the most common approaches to the solving few-shot problems. 

Recall that the goal of few-shot learning is to be able to learn to solve a new machine learning task given only a few labeled examples. In few-shot learning problems, we are given a small set of labeled examples for each class we would like to predict (the support set), as well as a larger set of unlabeled examples (the query set). We tend to refer to few-shot learning tasks as \(K\)-way, \(N\)-shot classification tasks, where \(K\) is the number of classes we would like to predict, and \(N\) is the number of labeled examples we are given for each class. 

When training a model to solve a few-shot learning task, we typically sample episodes from a large training dataset. An episode is a simulation of a few-shot learning task, where we sample \(K\) classes and \(N\) labeled examples for each class. Training a deep model by sampling few-shot learning episodes from a large training dataset is known as **episodic training**.

Here are the few-shot learning approaches covered in this tutorial:
1. **Metric-based few-shot learning**
2. **Optimization-based few-shot learning**


# Metric-Based Few-Shot Learning

Metric-based approaches to few-shot learning are able to learn an embedding space where examples that belong to the same class are close together according to some **metric**, even if the examples belong to classes that were not seen during training. 

![Metric-based learning](images/fsl/foundations/metric-based-learning.png)


<!-- Episodic training is essential to making metric-based few-shot models succeed in practice. Without episodic training, training a model using only $K$ examples for each class would result in poor generalization, and the model would not be able to generalize to new classes.  -->

At the center of metric-based few-shot learning approches is a similarity _metric_, which we will refer to as \(g_{sim}\). We use this similarity metric to compare how similar examples in the query set are to examples in the support set. After knowing how similar a query example is to each example in the support set, we can infer to which class in the support set the query example belongs to. Note that this is conceptually the same as performing a [nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search). 

This similarity comparison is typically done in the embedding space of some neural net model, which we will refer to as \(f_\theta\). Thus, during episodic training, we train \(f_\theta\) to learn an embedding space where examples that belong to the same class are close together, and examples that belong to different classes are far apart. This embedding model is sometimes also referred to as a _backbone_ model.

There are many different metric-based approaches to few-shot learning, and they all differ in how they define the similarity metric \(g_{sim}\), and how they use it to compare query examples to support examples as well as formulate a training objective.

Among the most popular metric-based approaches are Prototypical Networks {cite}`snell2017prototypical`, Matching Networks {cite}`vinyals2016matching`, and Relation Networks {cite}`sung2018relation`.

## Example: Prototypical networks

![Prototypical net](images/fsl/foundations/prototypical-net.png)

The figure above illustrates a 5-shot, 3-way classification task between tambourine (red), maracas (green), and djembe (blue). In prototypical networks, each of the 5 support vectors are averaged to create a prototype for each class (\(c_k\)). The query vector \(x\) is compared against each of the prototypes using squared eucldean distance. The query vector (shown as \(x\)) is assigned to the class of the prototype that it is most similar to. Here, the prototypes \(c_k\) are shown as black circles. 

> Prototypical networks {cite}`snell2017prototypical` work by creating a single embedding vector  for each class in the support set, called the **prototype**. The prototype for a class is the mean of the embeddings of all the examples in the support set for that class.

The prototype (denoted as \(c_k\)) for a class \(k\) is defined as: 

$$c_k = 1 / |S_k| \sum_{x_k \in S_k} f_\theta(x_k)$$

where \(S_k\) is the set of all examples in the support set that belong to class \(k\), \(x_k\) is an example in \(S_k\), and \(f_\theta\) is the backbone model we are trying to learn. 

After creating a prototype for each class in the support set, we use the euclidean distance between the query example and each prototype to determine which class the query example belongs to. We can build a probability distribution over the classes by applying a softmax function to the negated distances between a given query example and each prototype:

$$p(y = k | x_q) = \frac{exp(-d(x_q, c_k))}{\sum_{k'} exp(-d(x_q, c_{k'}))}$$

where \(x_q\) is a query example, \(c_k\) is the prototype for class \(k\), and \(d\) is the squared euclidean distance between two vectors.

## Prototypical Networks are Zero-Shot Learners too!

![Protonet ZSL](images/fsl/foundations/protonet-zsl.png)

When used for zero-shot learning (ZSL), prototypical networks don't require a labeled support set of examples for each class. Instead, the model is trained using a set of class metadata vectors, which are vectors that describe the characteristics of each class that the model should be able to classify. These metadata vectors are mapped to the same embedding space as the inputs using a separate model, \(g_\theta\), and used as the prototypes for each class. The prototypical network then uses the distances between the input and the class prototypes to make predictions about the input's category.

The prototypical network method can also be used for zero-shot learning. The method remains mostly the same as above. 
However, instead of relying on a support set \(S_k\) for each class \(k\), we are given some class metadata vector \(v_k\) for each class. 

The class metadata vector \(v_k\) is a vector that contains some information about the class \(k\), which could be in the form of a text description of the class, an image, or any other form of data.  During training, we learn a mapping \(g_\theta\) from the class metadata vector \(v_k\) to the prototype vector \(c_k\): \(c_k = g_\theta(v_k)\).

In this zero-shot learning scenario, we are mapping from two different domains: the domain of the class metadata vectors \(v_k\) (ex: text) and the domain of the query examples \(x_q\) (ex: audio). 

This means that we are learning two different backbone models that map to the **same embedding space**: \(f_\theta\) for the input query and \(g_\theta\) for the class metadata vectors.

# Optimization-Based Few-Shot Learning 

Optimization-based approaches focus on learning model parameters \(\theta\) that can easily adapt to new tasks, and thus new classes. The canonical method for optimization-based few-shot learning is Model-Agnostic Meta Learning (MAML) {cite}`finn2017model`,
and it's successors {cite}`li2017meta, sun2019mtl`. 

The intuition behind MAML is that some representations are more easily transferrable to new tasks than others. 

![Optimization-based FSL](images/fsl/foundations/opt-based-fsl.png)

For example, assume we train a model with parameters \(\theta\) to classify between (`piano`, `guitar`, `saxophone` and `bagpipe`) samples. 
Normally, we would expect that these parameters \(\theta\) would not be useful for classifying between instruments outside the training distribution, like `cello` and `flute`. The goal of MAML is to be able to learn parameters \(\theta\) that are useful not just for classifying between the instruments in the training set, but also are easy to adapt to new instrument classification tasks given a support set for each task, like `cello` vs `flute`, `violin` vs `trumpet`, etc.

In other words, if we have some model parameters \(\theta\), we want \(\theta\) to be adapted to new tasks using only a few labeled examples (a single support set) in a few gradient steps. 

The MAML algorithm accomplishes this by training the model to adapt from a starting set of parameters \(\theta\) to a new set of parameters \(\theta_i\) that are useful for a particular episode \(E_i\). This is performed for all episodes in a batch, eventually learning a starting set of parameters \(\theta\) that can be successfully adapted to new tasks using only a few labeled examples.

Note that MAML makes no assumption of the model architecture, thus the "model-agnostic" part of the method.

## The MAML algorithm

![MAML](images/fsl/foundations/maml.png)

> The MAML algorithm {cite}`finn2017model`. The starting model parameters are depcted as $\theta$, while the task-specific, fine-tuned parameters for tasks 1, 2, and 3 are depicted as $\theta_1^*$, $\theta_2^*$, and $\theta_3^*$, respectively. 

Suppose we are given a meta-training set composed of many few-shot episodes \(D_{train} = \{E_1, E_2, ..., E_n\}\), where each episode contains a support set and train set \(E_i = (S_i, Q_i)\). We can follow the MAML algorithm to learn parameters \(\theta\) that can be adapted to new tasks using only a few examples and a few gradient steps. 


Overview of the MAML algorithm {cite}`finn2017model`:
1. Initialize model parameters $\theta$ randomly, choose a step sizes \(\alpha\) and \(\beta\).  
2. **while** not converged **do**

    3. Sample a batch of episodes (tasks) from the training set \(D_{train} = \{E_1, E_2, ..., E_n\}\)
    4. **for** each episode \(E_i\) in the batch **do**

        5. Using the current parameters \(\theta\), compute the gradient of the loss \(L_if(\theta)\) for episode $E_i$.

        6. Compute a new set of parameters \(\theta_i\) by fine-tuning in the direction of the gradient w.r.t. the starting parameters \(\theta\): 
        $$\theta_i = \theta - \alpha \nabla_{\theta} L_i$$

    7. Using the fine-tuned parameters \(\theta_i\) for each episode, make a prediction and compute the loss \(L_{i}f(\theta_i)\).

    8. Update the starting parameters \(\theta\) by taking a gradient step in the direction of the loss we computed with the fine-tuned parameters \(L_{i}f(\theta_i)\):

        $$\theta = \theta - \beta \nabla_{\theta} \sum_{E_i \in D_{train}}L_i f(\theta_i)$$


At inference time, we are given a few-shot learning task with support and query set \(E_{test} = (S_{test}, Q_{test})\). We can use the learned parameters \(\theta\) as a starting point, and follow a process similar to the one above to make a prediction for the query set \(Q_{test}\):  

1. Initialize model parameters \(\theta\) to the learned parameters from meta-training.
2. Compute the gradient \(\nabla_{\theta} L_{test} f(\theta)\) of the loss \(L_{test}f(\theta)\) for the test episode \(E_{test}\).
3. Similar to step 6 of the training algorithm above, compute a new set of parameters \(\theta_{test}\) by fine-tuning in the direction of the gradient w.r.t. the starting parameters \(\theta\). 
4. Make a prediction using the fine-tuned parameters \(\theta_{test}\): \(\hat{y} =(\theta_{test})\).
