<!---
Para otros años, mirar también este código: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
-->

> Submission of the assignments is due to Feb. 21th

# Prototypical Networks

This assignment is part of the HW2 exercise from the [Stanford CS330 Autumn 2020 Homework 2 Model-Agnostic Meta-Learning and Prototypical Networks](http://cs330.stanford.edu/fall2020/index.html) course.

In this assignment we will experiment with prototypical networks [@snell2017prototypical], training a model for \(K\)-shot, \(N\)-way classification. For this, we will work with the Omniglot dataset [@omniglot], which includes 1623 handwritten characters from 50 different alphabets. You can see some samples of this dataset below:

![Omniglot](images/fsl/omniglot.jpg)

As discussed in the [theory contents](https://pertusa.github.io/ap/fsl/#metric-based-few-shot-learning), the basic idea of prototypical networks resembles nearest neighbors to class prototypes. They compute the prototype of each class using a set of support examples and then calculate the distance between the query example and each the prototypes. The query example is classified based on the label of the prototype it’s closest to, as can be seen in the following figure:

![Prototypical](images/fsl/prototypical.jpg)

> If you think that you didn't fully understand prototypical networks after reading the provided materials, please see also [this video](https://www.youtube.com/watch?v=rHGPfl0pvLY), which clearly explains their foundations.

## Exercise 1

First, open [this Colab notebook](https://colab.research.google.com/drive/1Ah1Os8TAItF42rLtAINNJfaqDJYXRJ7X?usp=sharing) which contains some code to be completed with **TODO** marks. You need to save a copy of this notebook to your Google Drive in order to make edits, and then upload the final `.ipynb` to Moodle.

1. Please fill in the data processing parts in the function `run_protonet`, which should also call the data generator provided. The sampled batch is partitioned into support, i.e. the per-task training data, and query, i.e. the per-task test datapoints. The support will be used to calculate the prototype of each class and query will be used to compute the distance to each prototype. 

2. Fill in the function called `ProtoLoss` which takes the embeddings of the support and query examples, as well as the one-hot label encodings of the queries, and computes the loss and prediction accuracy based on the main algorithm of the prototypical networks.

3. Run the cell `run protonet(‘./omniglot resized/’)` and check the results. The average test accuracy for the default parameters after the training process should be around 0.84. 

4. Try with \(K = 4, 6, 8, 10\) at _meta-test_ time. Compare the _meta-test_ performance by analyzing the _meta-test_ accuracies over different choices of \(K\) and discuss the results at the end of the notebook.

# Few-shot learning using OpenAI GPT and Google Gemini

We know many of you are GPT fans, so we are going to make an assignment using this tool. 

As we have seen [in the theory](https://pertusa.github.io/ap/fsl/#openai-gpt-3), very Large Language Models (LLM) can perform few-shot learning with minimal steps. 

Let's test an example by promping the following input to [ChatGPT-3.5](https://chat.openai.com/):

```
Input: Subpar acting. 
Sentiment: Negative 
Input: Beautiful film. 
Sentiment: Positive 
Input: Amazing. 
Sentiment:
```

> Hint: If you want to insert a new line without sending the prompt, simultaneously press shift + enter.

Run this prompt and check the result. We have just created a sentiment analysis classifier without any line of code, although of course it has some limitations. By using the [ChatGPI API](https://help.openai.com/en/articles/7039783-how-can-i-access-the-chatgpt-api) you can even integrate your classifier into a webpage or an app.

This is a simple example, but making reliable prompts for accurate few-shot learning classifiers is not straighforward. Please, have a look to this paper first:
https://arxiv.org/abs/2102.09690. You can see in Fig. 4 how the order and balance of the positive/negative examples can affect the results.

The goal of this assignment is assesssing your understanding of how to effectively employ few-shot learning techniques on GPT and Gemini[@gemini] models. 

> [Google Gemini](https://gemini.google.com/app) is a very recent LLM similar to ChatGPT. One important difference from ChatGPT is that Gemini can also access to the web to search for updated information. 

## Exercise 2

In this second exercise, we are going to make a few-shot classifier to classify between *flu* (_gripe_), *cold* (_resfriado_), or *None* based on a description of the symptoms. This classifier can be trained on a few examples of symptoms and their corresponding label. Then, given a new set of symptoms, it could predict the possible pathology.

> You can find information to help distinguishing between flu and cold in [this link](https://www.cdc.gov/flu/symptoms/coldflu.htm).

Make a comparison between GPT3 and Gemini, discussing how the ordering and balance of the different classes affect the results. 

Finally, submit a PDF file (via Moodle) with the experiments you made and the conclusions.

### Assessment criteria

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
