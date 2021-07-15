import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

sequence_length = 100
BATCH_SIZE = 256
EPOCHS = 30

FILE_PATH = "data.txt"
BASENAME = os.path.basename(FILE_PATH)

# Read the data
text = open(FILE_PATH, encoding="utf-8").read()

# Remove capital letters
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", punctuation))

# Print some stats
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)
print("\n")

# Dictionaries for conversion
char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

# Save dictionaries
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

# convert all text to int
encoded_text = np.array([char2int[c] for c in text])
print("encoded_text length:", len(encoded_text))
print("\n")
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

# print first 5 chars
for char in char_dataset.take(8):
    print("First 5 chars:")
    print(char.numpy(), int2char[char.numpy()])
    print("\n")

# Build sequences
sequences = char_dataset.batch(2 * sequence_length + 1, drop_remainder=True)

# print sequences
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))



def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample) - 1) // 2):
        input_ = sample[i: i + sequence_length]
        target = sample[i + sequence_length]
        # Extend the data set with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

# Prep inputs and targets
dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)


# Print first 2 samples
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

# Repeat, shuffle and batch the dataset
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax")
])

# model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Make sure the results folder exists. If it doesn't, create one.
if not os.path.isdir("results"):
    os.mkdir("results")

# Train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)

# Save the model
model.save(f"results/{BASENAME}-{sequence_length}.h5")


