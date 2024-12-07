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
```
## Dataset
The IMDB reviews dataset is used for binary sentiment classification (positive or negative). It is downloaded and prepared using TensorFlow Datasets (`tensorflow_datasets`).

## Model Architecture
The model uses the following layers:
1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **Bidirectional LSTM**: Captures dependencies in both directions of the text.
3. **Dense Layers**: For feature transformation and binary classification.

Key parameters:
- Vocabulary size: `10000`
- Sequence length: `120`
- Embedding dimensions: `16`
- LSTM units: `32`
- Dense layer units: `6`

## How to Use
1. Clone the repository:

```bash
git clone 
```
2. Run the script:
```bash
 
```
3. Training and evaluation will be performed, and plots of accuracy and loss will be displayed.

## Training Performance
The script includes a utility function to visualize the training and validation performance over epochs:
- Training and Validation Accuracy
- Training and Validation Loss

## Directory Structure
```bash
.
├── data/               # Dataset directory (downloaded by TensorFlow Datasets)
├── sentiment_analysis.py # Main script
├── README.md           # Project documentation
```
