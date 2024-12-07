import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Download and prepare the dataset
imdb = tfds.load('imdb_reviews', as_supervised=True, data_dir='./data', download=False)
train_dataset, test_dataset = imdb['train'], imdb['test']

# Parameters (Vectorization & Padding)
VOCAB_SIZE = 10000
MAX_LENGTH = 120
PADDING_TYPE = 'pre'
TRUNC_TYPE = 'post'

def preprocessing_fn(dataset):
    '''Generates padded sequences from a tf.data.Dataset'''
    # Apply the text vectorization
    dataset_sequences = dataset.map(lambda text, label: (vectorize_layer(text), label))
    # Pull all elements in a single ragged batch
    dataset_sequences = dataset_sequences.ragged_batch(batch_size=dataset_sequences.cardinality())
    # Single batch tensor
    sequences, labels = dataset_sequences.get_single_element()
    # Pad the sequences
    padded_sequences = tf.keras.utils.pad_sequences(
        sequences.numpy(),
        maxlen=MAX_LENGTH,
        truncating=TRUNC_TYPE,
        padding=PADDING_TYPE,
    )
    # Convert back to a tf.data.Dataset
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    # Combine the reviews and labels
    dataset_vectorized = tf.data.Dataset.zip(padded_sequences, labels)
    return dataset_vectorized

def plot_loss_acc(history):
    '''Plot the training and validation loss and accuracy as a function of epochs'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].plot(epochs, acc, 'b', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].legend()
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    ax[1].plot(epochs, loss, 'b', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].legend()
    ax[1].set_title('Training and validation loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')

    plt.show()



# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
# Get the string inputs and integer outputs of the training set
train_reviews = train_dataset.map(lambda review, label: review)
# Generate the vocab based only on the training set
vectorize_layer.adapt(train_reviews)
# Delete train_reviews it's no longer needed
del train_reviews

# Preprocess the train and test data
train_dataset_vectorized = train_dataset.apply(preprocessing_fn)
test_dataset_vectorized = test_dataset.apply(preprocessing_fn)

# View training sequences and its labels
for ex in train_dataset_vectorized.take(2):
    print(ex)
    print()

# Dataset Optimization
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = (train_dataset_vectorized
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       )
test_dataset_final = (test_dataset_vectorized
                      .cache()
                      .batch(BATCH_SIZE)
                      .prefetch(PREFETCH_BUFFER_SIZE))

# Model
EMBEDDING_DIMS = 16
LSTM_DIMS = 32
DENSE_DIM = 6
NUM_EPOCHS = 10

# Model Definition with LSTM
model_lstm = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIMS),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIMS)),
    tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Set the training parameters
model_lstm.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
# Print the model summary
model_lstm.summary()

# Train the model
history = model_lstm.fit(train_dataset_final,epochs=NUM_EPOCHS,validation_data=test_dataset_final)

# Plot the accuracy and loss history
plot_loss_acc(history)