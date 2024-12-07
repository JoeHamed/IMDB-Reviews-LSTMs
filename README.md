# IMDB Reviews Sentiment Analysis

This project implements a sentiment analysis model using the IMDB movie reviews dataset. It preprocesses the data using TensorFlow's `TextVectorization` layer, trains an LSTM-based neural network, and visualizes the training performance.

## Features
- Uses **IMDB movie reviews dataset** from TensorFlow Datasets.
- Text preprocessing includes **tokenization**, **padding**, and **truncation**.
- Built with a **Bidirectional LSTM** model for better handling of sequence data.
- Visualizes **accuracy** and **loss trends** during training.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- TensorFlow Datasets

Install the required packages using pip:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib
