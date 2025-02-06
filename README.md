# Siamese Neural Network - Fish Identification Project

## Overview

This project focuses on identifying fish species using a Siamese Neural Network (SNN). The network is designed to compare pairs of images and determine their similarity based on learned feature representations. The model is trained on a carefully curated dataset to distinguish between 23 different fish species.

## src

The src folder contains the main source code for the project. It includes all the scripts required for training, testing, and running the Siamese Neural Network.

Key features:

* Contains a small sample dataset to test the code and verify functionality.
* Modular design with clearly separated components for data handling, model training, evaluation, and utilities.
* Well-documented scripts for easy understanding and customization.

## Dataset

### Source

The dataset for this project is sourced from the Fish4Knowledge repository. It is a comprehensive dataset of underwater fish images acquired from live video streams. The dataset includes:

* 27,370 manually labeled fish images.
* 23 unique clusters, each representing a specific fish species.
* Images grouped based on morphological features such as fin shapes, sizes, and anatomical distinctions, as defined by marine biologists.

### Dataset Characteristics

* The dataset exhibits class imbalance, with some fish species being significantly more frequent than others.
* Images are captured in real-world underwater conditions, introducing variability in lighting, orientation, and movement.

### Dataset Links

* Fish4Knowledge Homepage: [Homepage Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/)
* Download Dataset: [Fish4Knowledge Dataset](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/)

## Web Application (Flask)

### Status

The web application is currently under development and is not yet integrated with the main src code.

### Features

* Web application for Testing: Supports testing of a single image using a pretrained Siamese Neural Network model.
* Similarity Analysis: The API evaluates the similarity score (distance) between the input image and reference images, providing the best match based on the SNN's predictions.

### Planned Features

* Full integration with the main codebase for seamless testing and evaluation.
* Enhanced visualization and interactive analysis of similarity scores and predictions.
* Support for multi-image detection and classification using a combination of object detection and the SNN.

## How to Use

* Clone the repository and navigate to the src folder.
* Use the provided sample dataset for testing, or download the full Fish4Knowledge dataset for complete functionality.
* Run the training or testing scripts to evaluate the SNN on fish species identification.
