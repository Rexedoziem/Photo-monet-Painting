# Photo-monet-Painting
This repository contains the code for the Photo-to-Monet Challenge on Kaggle. The challenge aims to train a CycleGAN model to generate Monet-style paintings from real-world photographs.
Challenge Description

The Photo-to-Monet Challenge is part of the Kaggle Competitions platform. The goal of the challenge is to create an algorithm that can transform photographs into Monet-style paintings using the CycleGAN architecture.

Participants are provided with a dataset of photographs and Monet-style paintings. The task is to train a model that can learn the mapping between the two domains and generate high-quality Monet-style paintings from input photographs.
# Dataset

The dataset for the Photo-to-Monet Challenge consists of two main components:

    Photographs: A collection of real-world photographs. These images serve as the input domain for the model.

    Monet-style Paintings: A set of Monet-style paintings created by the artist Claude Monet. These images represent the target domain that the model needs to learn to generate.

The dataset is provided in a preprocessed format, with images in a standardized size and format.
# Repository Structure

The repository is organized as follows:

    config.py: Configuration file containing hyperparameters and settings for the model training.

    train.py: The main training script that loads the dataset, initializes the CycleGAN model, and performs the training loop.

    models.py: Contains the implementation of the Generator and Discriminator models used in the CycleGAN architecture.

    utils.py: Utility functions for data loading, model initialization, and other helper functions.

    image_transforms.py: Custom image transforms and data augmentation functions.

    README.md: This readme file.

# Getting Started

To get started with the Photo-to-Monet Challenge, follow these steps:

    Clone this repository to your local machine:

bash

    git clone https://github.com/Rexedoziem/Photo-monet-Painting.git

Install the required dependencies by running the following command:

Download the dataset for the challenge from the Kaggle competition page. Place the dataset files in the appropriate directory within the repository.

Modify the config.py file to adjust the hyperparameters and settings according to your requirements.

Start the training process by running the train.py script:

bash

    python train.py

    Monitor the training progress and evaluate the model's performance.

# Contributing

If you'd like to contribute to the Photo-to-Monet Challenge, please follow these guidelines:

    Fork the repository on GitHub.

    Create a new branch for your features or fixes.

    Make your changes and commit them with descriptive messages.

    Push your changes to your forked repository.

    Submit a pull request to the main repository, explaining your changes and their purpose.

    Ensure that your code follows the repository's coding style and conventions.

# License

This project is licensed under the MIT License.
# Acknowledgments

    This challenge is hosted on Kaggle as part of the community-driven competitions platform.

    The dataset used in this challenge is provided by Kaggle.

    We thank the organizers and contributors of the Photo-to-Monet Challenge for providing the opportunity to participate and learn.
