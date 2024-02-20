# Deep Reinforcement Learning
!!! danger
    - This is work in progress. The contents of this page are not final and, therefore, it is not recommended to start working on its contents yet.


> Read this page and follow the instructions before **February, 28th**.

!!! info "Table of contents"
    - [Introduction](#introduction)
    - [Self-study](#self-study)
    - [Materials for coding practice](#materials-for-coding-practice)

# Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards, which it uses to improve its decision-making over time.

Deep Reinforcement Learning (DRL) is an extension of RL that incorporates deep learning techniques. Instead of relying on handcrafted features or representations of the environment, DRL algorithms use deep neural networks to directly learn from raw sensory input, enabling them to tackle more complex tasks and achieve superior performance.

DRL adds more advantages by enabling high-dimensional or infinite state and action spaces through the use of universal function approximators such as neural networks. This allows DRL algorithms to handle a wide range of tasks that were previously infeasible with traditional RL methods.

## Fundamentals of Reinforcement Learning

RL is a paradigm in machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, where the model learns from labeled data, or unsupervised learning, where the model identifies patterns in unlabeled data, RL takes a very different approach. Instead of learning from explicit examples, here the agent learns through a process of *trial and error*, receiving feedback in the form of rewards based on its actions.

### Understanding the Players

In this setup, we have two main players: the **agent** and the **environment**. The agent is the learner, the one making decisions, while the environment is everything the agent interacts with. Together, they create a dynamic system where actions lead to consequences.

### The Decision-Making Process

At each step of the process, the agent observes the current **state** of the environment, selects an **action** based on its strategy or policy, executes that action, and receives a **reward** or penalty from the environment. This feedback loop is fundamental to the learning process.

### Key Concepts

#### States, Actions, and Rewards

- **State (S)**: The representation of the environment at a given moment.
- **Action (A)**: The decision made by the agent based on the current state.
- **Reward (R)**: The feedback received from the environment after taking an action.

#### Policies and Value Functions

- **Policy (π)**: The strategy or rule the agent uses to select actions in different states.
- **Value Function (V)**: The expected cumulative reward an agent can achieve from a particular state.
- **Q-Value Function (Q)**: The expected cumulative reward an agent can achieve by taking a particular action in a given state.

### Paradigms in Deep Reinforcement Learning

The ultimate goal of reinforcement learning is to find an optimal strategy or policy that maximizes the cumulative rewards over time. To attain this goal, there are different types of methods.

#### Value-Based Methods

Value-based methods aim to learn the optimal value function, which estimates the expected gain of taking an action in a given state and following a specific policy thereafter. Deep Q-Networks (DQN) is a prominent example of a value-based method, where a deep neural network is trained to approximate the Q-function.

#### Policy-Based Methods

Policy-based methods directly learn the policy, i.e., the mapping from states to actions, without explicitly estimating the value function. This approach can be more effective in high-dimensional or continuous action spaces. Examples include the REINFORCE algorithm and its variants, which optimize the neural network parameters to maximize expected rewards from a given state.

#### Actor-Critic Methods

Actor-Critic methods combine aspects of both value-based and policy-based approaches. They maintain two neural networks: one (the actor) learns the policy, while the other (the critic) learns the value function. The critic provides feedback to the actor by evaluating the chosen actions, helping to guide the policy towards better decisions.

## Self-study

Now is the time to delve into certain aspects of DRL before the in-person class. You must ensure a clear understanding of:

1. The **RL paradigm** and **its fundamental elements** (state, action, policy, reward, value...).
2. **DQN** and **REINFORCE** algorithms, and their differences.
3. The **concept** behind **actor-critic** approaches.

:octicons-book-24: The recommended lecture is the book [**Understanding Deep Learning**](https://udlbook.github.io/udlbook/) by Simon J.D. Prince, specifically the parts concerning DRL:

- Introduction at Section 1.3 *(Book pp. 11-12; PDF pp. 25-26)*
- Chapter 19: Reinforcement Learning *(Book pp. 373-412; PDF pp. 387-398)*
    - *You can skip sections 19.3.1, 19.5.1, 19.7*

!!! note "You are encouraged to consult any other materials. These might include, but are not limited to:"
    - **Online resources**:
        - **RL/DRL**: [OpenAI Spinning Up: Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
        - **DQN**: [CS234 Notes - Lecture 6: CNNs and Deep Q Learning](https://web.stanford.edu/class/cs234/CS234Win2019/slides/lnotes6.pdf)
        - **REINFORCE**: [REINFORCE – A Quick Introduction (with Code)](https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code/) 

    - **Courses**:
        - [RL Course](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) by David Silver (Google DeepMind) on Youtube
        - [CS 285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) (Berkeley)

    - **Books**:
        - "[Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)" by Richard S. Sutton and Andrew G. Barto

## Materials for coding practice

During the in-person practice session we will solve a classic control problem called *CartPole* using DRL. To simulate the scenario, we will work with the framework **Gymnasium**. Below you will find a Google Colaboratory notebook that introduces this working environment. You must read and understand this notebook before the session.

- Introduction to Gymnasium framework [[link to colab]](https://colab.research.google.com/drive/1ETiv5bo6db5F0xa3QkH9q_m01jLL0Rkm?usp=sharing)