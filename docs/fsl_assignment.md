# Few and Zero-Shot Learning assignment

<!---
Para otros años, mirar también este código: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
-->

> This assigment must be submitted by **February 21th** using Moodle

## Prototypical Networks

The following exercise is (a tiny) part of the HW2 task from the [Stanford CS330 Autumn 2020 Homework 2 Model-Agnostic Meta-Learning and Prototypical Networks](http://cs330.stanford.edu/fall2020/index.html) course.

In this exercise we will experiment with prototypical networks[@snell2017prototypical], training a model for \(K\)-shot, \(N\)-way classification. For this, we will work with the Omniglot dataset[@omniglot], which includes 1623 handwritten characters from 50 different alphabets. You can see some samples of this dataset below:

![Omniglot](images/fsl/omniglot.jpg)

As discussed in the [theory contents](https://pertusa.github.io/ap/fsl/#metric-based-few-shot-learning), the basic idea of prototypical networks resembles nearest neighbors to class prototypes. They compute the prototype of each class using a set of support examples and then calculate the distance between the query example and each the prototypes. The query example is classified based on the label of the prototype it’s closest to, as can be seen in the following figure:

![Prototypical](images/fsl/prototypical.jpg)

> If you think that you didn't fully understand prototypical networks after reading the provided materials, please see also [this video](https://www.youtube.com/watch?v=rHGPfl0pvLY), which clearly explains their foundations.

### Exercise 1 (60%)

First, open [this Colab notebook](https://colab.research.google.com/drive/1Ah1Os8TAItF42rLtAINNJfaqDJYXRJ7X?usp=sharing) which contains some code to be completed with **TODO** marks. You need to save a copy of this notebook to your Google Drive in order to make edits, and then upload the final `.ipynb` to Moodle.

1. Please fill in the data processing parts in the function `run_protonet`, which should also call the data generator provided. The sampled batch is partitioned into support, i.e. the per-task training data, and query, i.e. the per-task test datapoints. The support will be used to calculate the prototype of each class and query will be used to compute the distance to each prototype. 

2. Fill in the function called `ProtoLoss` which takes the embeddings of the support and query examples, as well as the one-hot label encodings of the queries, and computes the loss and prediction accuracy based on the main algorithm of the prototypical networks.

3. Run the cell `run protonet(‘./omniglot resized/’)` and check the results. The average test accuracy for the default parameters after the training process should be around 0.84. 

4. Try with \(K = 4, 6, 8, 10\) at _meta-test_ time. Compare the _meta-test_ performance by analyzing the _meta-test_ accuracies over different choices of \(K\) and discuss the results at the end of the notebook.

> As you have seen, the provided `ipynb` has a complicated code since it was written for Tensorflow. If you really want to train  Prototypical networks for other projects, I recommend to have a look at [this Pytorch](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb) notebook.

## Few-shot learning using OpenAI GPT

We know many of you are GPT fans, so we are going to make an assignment using this tool. 

As we have seen [in the theory](https://pertusa.github.io/ap/fsl/#openai-gpt-3) contents, very Large Language Models (LLM) can perform few-shot learning with minimal steps. 

Let's test an example by promping the following input to [ChatGPT-3.5](https://chat.openai.com/):

```
Input: Subpar acting. 
Sentiment: Negative
Input: Beautiful film. 
Sentiment: Positive
Input: Amazing. 
Sentiment:
```

> Hint: In GPT, if you want to insert a new line without sending the prompt, simultaneously press shift + enter

Run this prompt and check the result. We have just created a sentiment analysis classifier without any line of code, although it may have  limitations in more complex scenarios. By using the [ChatGPI API](https://help.openai.com/en/articles/7039783-how-can-i-access-the-chatgpt-api) you can even integrate your sentiment classifier into a webpage or an app.

This is a simple example, but making reliable prompts for accurate few-shot learning sometimes require additional work. For example, have a look at [this paper](https://arxiv.org/abs/2102.09690)[@calibratellm]. You can see in Fig. 4 how the order and balance of the positive/negative examples can affect the results.

> In the current GPT3.5 version, behaviour is a bit different than in the paper. For example, prompts with N/A are not accepted

The goal of the following exercise is assesssing your understanding of how to effectively employ few-shot learning techniques on GPT.

> We could also have used [Google Gemini](https://gemini.google.com/app)[@gemini]. This is a very recent LLM similar to ChatGPT, but Gemini can also access the web to search for updated information. Therefore, since it uses external data, it is not suitable for our few-shot learning scenario.

### Exercise 2 (40%)

In this second exercise, we are going to make a few-shot classifier to classify between *Rock* and *Reggaeton* genres from a short part of song lyrics. To achieve this goal, the classifier must be trained on a few examples of lyrics and their corresponding labels. Then, given new lyrics, it should ideally predict the song genre.

An example prompt could be: 
```
Input: No pienses que estoy muy triste
Si no me ves sonreír
Es simplemente despiste
Maneras de vivir.
Output: Rock

Input: Me acostumbré al sour, ya no patea
Me llegan a casa, no se capean
Solo modelos como Barea
Multiplicar cienes es la tarea.
Output: Reggaeton
```

> You can find more lyrics examples [in this link](https://www.letras.com/)

The goal is to effective prompts, and check if ordering of the samples and balance of the classes may affect the results. For our few-shot scenario, try with \(N=6\) labeled samples for each of the 2 classes.

Once done, please submit a PDF file (via Moodle) with the experiments you made and the conclusions.

#### Assessment criteria

- **Prompt Engineering:** Students will be evaluated on their ability to engineer effective prompts that leverage the few-shot examples. This includes the clarity of the prompt, the relevance of the examples to the test case, and the prompt's ability to guide the model towards the desired output.
- **Model Interaction:** Students may need to iteratively refine their prompts based on the model's responses, demonstrating an understanding of how different prompt structures influence the outcome.
- **Critical Analysis:** In addition to generating outputs, students should critically analyze the model's performance, identifying any biases, errors, or limitations in the generated responses.

<!--
### Task 1: Scientific text Summarization
**Objective:** Employ few-shot learning to enable a GPT model to summarize academic (scientific paper) abstracts.
- **Few-Shot Examples:** Provide 3 examples of academic abstracts along with their concise summaries.
- **Test:** Given an academic abstract not seen by the model, generate a prompt that leads the model to produce a coherent and concise summary.

### Task 2: Code Generation from Descriptions
**Objective:** Use few-shot learning to teach a GPT model to generate Python code snippets from natural language descriptions.
- **Few-Shot Examples:** Supply 4 examples of natural language descriptions of programming tasks alongside their corresponding Python code snippets.
- **Test:** Provide a new, detailed description of a programming task, and devise a prompt that will guide the model to generate the appropriate Python code.

<!--
### Task 4: Translation
**Objective:** Adapt a GPT model for language translation tasks using a few-shot approach.
- **Few-Shot Examples:** Offer 5 pairs of sentences, each in English and its translation in Spanish.
- **Test:** Give a sentence in English and ask the student to construct a prompt that encourages the GPT model to translate it into Spanish accurately, leveraging the few-shot examples.

### Task 5: Question Answering
**Objective:** Train a GPT model to answer domain-specific questions with few-shot examples.
- **Few-Shot Examples:** Provide 5 question-answer pairs in a specialized field (e.g., biology, computer science).
- **Test:** Present a new, complex question in the same domain and have the student create a prompt that would enable the GPT model to use the few-shot examples to answer accurately.



### Task 3: Ethical Judgment
**Objective:** Guide a GPT model to make ethical judgments in hypothetical scenarios using few-shot learning.
- **Few-Shot Examples:** Share 3-4 scenarios involving ethical dilemmas, each with a reasoned judgment on why a particular action is ethically sound or unsound.
- **Test:** Describe a new ethical scenario and instruct the student to formulate a prompt that aids the model in providing an ethical judgment, drawing on the reasoning from the examples.
-->
