# Hacking-the-Capsule-Network

This repository is contains Hands on guide to learning PyTorch and developing Machine learning models using it. This repo describes Capsule Networks and its fundamental aspects along with implementation in PyTorch.

In case if the notebook is not loading, please click on the link below:
<br/>
[capsule networks.ipynb](https://nbviewer.jupyter.org/github/PyExtreme/Hacking-the-Capsule-Network/blob/master/capsule%20networks.ipynb)

## Introduction - Capsule Networks

A capsule network takes input an image and finds out what kind of object is present and what are the instatiation parameters(rotation, thickness, etc).

* **Why capsule networks?** 

Pooling allows a degree of translational invariance (it can recognize the same object in a somewhat different location) and allows a larger number of feature types to be represented. The main disadvantage of Pooling is that pooling can't explore the spatial relationship and provides invariance instead of equivariance i.e translation of input features results in an equivalent translation of outputs. In brief, CNN contains pooling layer which can't capture spatial relationship like poses of an entity. But capsule is designed in such a way that it can easily capture these features.


**A capsule network is composed of many capsules**

## Capsule

A capsule is a group of neurons whose activity vectors represent various instantiation parameters of an object. For classifying an entity to class, there should be an agreement between multiple capsules. Basically, this process goes like this:

* Active capsules at the lower level makes prediction for the instantiation parameters of higher level capsules.Tthe lower level capsules represents lower level features like presence of rectangle or triangle whereas the higher level features represents boat or house.
* When multiple capsules of a lower level come to an agreement, the capsules at higher level become active. Hence, capsule nets are able to establish a hierarchy of features between various levels of capsules. For eg, from an image of boat, capsnet can capture the presence of a triangle over a rectangle entity as a whole and as we move down the lower levels, it can capture features like presence of rectangle and a triangle.
* The probability of presence of entity is determined by the length of the activity vectors and the vector's orientation tells about the properties of an entity.
The length of the activity vector is fed into a non-linear transformation called the squashing function so that the length of output vector becomes less than 1 because it is used to represent probability.

The main component of capsule network is the **routing by agreement** strategy by which capsule of different levels interact with each other.

## Routing By agreement

This is the pivotal part of capsule network. Initially when the feature maps have been produced by the CNN layer, the capsules at the lower level try to predict the output of the capsules at the higher level. When the capsules at the lowel level come upon an agreement over the presence of an entity at higher level, the capsule corresponding to that entity becomes active. Hence, other capsules are rejected and this is instrumental in removing noise from the model. Therefore, when there is a strong agreement between the capsules at the lower levels for the activity of the capsules at the higher level, the capsules at the lower level are routed to the capsules at higher level.


## Margin Loss for capsule networks

![image](https://user-images.githubusercontent.com/18027903/51508068-c9ca3a00-1e19-11e9-8688-17c2fef643dc.png)

## Dataset

The dataset used is Fashion MNIST dataset containing 60000 images of all 10 different classes.
The link to dataset is [here](https://www.kaggle.com/zalando-research/fashionmnist)
