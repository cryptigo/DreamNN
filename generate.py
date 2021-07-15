import numpy as np
import pickle
from string import punctuation
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import os
import argparse
from datetime import date


parser = argparse.ArgumentParser(description='Generates a dream fanfiction',
                                 prog='generate.py',
                                 usage='%(prog)s [options] seed')
parser.add_argument('Seed',
                    metavar='seed',
                    type=str,
                    help='The string to generate from')

prog_args = parser.parse_args()
def Main():
    sequence_length = 100
    FILE_PATH = "data.txt"
    BASENAME = os.path.basename(FILE_PATH)
    gen_seed = prog_args.Seed

    # process seed
    # - remove punctuation, make lower
    gen_seed = gen_seed.lower()
    gen_seed = gen_seed.translate(str.maketrans("", "", punctuation))


    # load vocab dict
    char2int = pickle.load(open(f"data/dict/{BASENAME}-char2int.pickle", "rb"))
    int2char = pickle.load(open(f"data/dict/{BASENAME}-int2char.pickle", "rb"))
    vocab_size = len(char2int)

    # Build model
    model = Sequential([
        LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(vocab_size, activation="softmax"),
    ])

    # load weights
    model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")

    s = gen_seed
    n_chars = 1000

    generated = ""

    for i in tqdm.tqdm(range(n_chars), "Generating text"):
        # Make input seq
        X = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(gen_seed):
            X[0, (sequence_length - len(gen_seed)) + t, char2int[char]] = 1

        # predict next char
        predicted = model.predict(X, verbose=0)[0]
        # convert vector to an int
        next_index = np.argmax(predicted)
        # convert the int to a char
        next_char = int2char[next_index]
        # add char to results
        generated += next_char
        gen_seed = gen_seed[1:] + next_char

    print("Seed:", s)
    print("Generated text:")
    print(generated)
    
    curr_time = date.today()
    
    f = open(f"data/generated-{curr_time}.txt", "w")
    f.write(generated)
    f.close()

if __name__ == '__main__':
    Main()
    
