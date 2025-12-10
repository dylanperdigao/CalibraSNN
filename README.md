# CalibraSNN: Fair and Calibrated Convolutional Spiking Neural Network for High-Stakes Industry Applications

[[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13546087.svg)](https://doi.org/10.5281/zenodo.13546087) ](https://doi.org/10.5281/zenodo.17253972)

Source code of the paper entitled "CalibraSNN: Fair and Calibrated Convolutional Spiking Neural Network for High-Stakes Industry Applications" published at "IEEE Access" journal.

## Paper Abstract

Real-world problems are often embedded in highly imbalanced contexts, where traditional machine learning algorithms struggle to achieve both strong performance and fairness. In high-stakes industries, incorporating sensitive attributes can enhance performance, but often at the cost of fairness. However, while artificial neural networks typically have the drawback of high energy usage, spiking neural networks (SNNs) offer a promising alternative because of their energy efficiency, in addition to their performance that is frequently hindered by their sensitivity to the hyperparameters. Our research introduces a fair and calibrated convolutional SNN modeling framework, CalibraSNN, designed for constrained, real-world problems by addressing the challenges in performance, fairness, energy usage, and sensitivity to hyperparameters. We evaluate our approach using the Bank Account Fraud dataset suite, which comprises real-world data and the business constraint of maintaining a false positive rate below 5%. Our results show up to 65% recall in fairly classifying fraudulent behaviors with competitive energy usage and power demand towards non-spiking approaches. The greener model is able to demand only 8 watts of power and use 30 joules with a recall of 54%. 

## Installation

To install the required packages, run the following command:
```sh
pip install -r requirements.txt
```
Download the six Variant of the Bank Account Fraud (BAF) Dataset and extract the parquet files to the data folder.

## Dataset

The Bank Account Fraud (BAF) dataset is a synthetic dataset based on real-world data that simulates bank account opening applications. The dataset contains 6 parquet files, each representing a different variant of the dataset (Base, Variant I, Variant II, Variant III, Variant IV, and Variant V). It contains 30 features and a binary target variable indicating whether the application is fraudulent or not.

## Repository Structure

The repository is structured as follows:

- `data`: Contains the Bank Account Fraud dataset.
- `images`: Contains the images used in this README file.
- `src`: Contains the source code of the project.

## Bibtex

To cite this work, use the following bibtex entry:
```bibtex
@article{11283005,
  author={Perdigão, Dylan and Antunes, Francisco and Silva, Catarina and Ribeiro, Bernardete},
  journal={IEEE Access}, 
  title={CalibraSNN: Fair and Calibrated Convolutional Spiking Neural Network for High-Stakes Industry Applications}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Neurons;Artificial intelligence;Spiking neural networks;Energy efficiency;Optimization;Mathematical models;Ions;Computational modeling;Power demand;Hardware;Spiking Neural Network;Neuromorphic Computing;Neural Network Calibration;Imbalanced Data;Fair ML;Green AI;Low-Power ML;High-Stakes Industries},
  doi={10.1109/ACCESS.2025.3641389}
}
```
Issues

This code is imported and adapted from the original research repository. Consequently, the code may contain bugs or issues. If you encounter any issues while running the code, please open an issue in the repository.
