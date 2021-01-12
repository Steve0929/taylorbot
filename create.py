from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

maxlen = 40
diversity = 1.0
#Prepare the data
with io.open("ts_songs.txt", encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
total_chars = len(chars)

# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model('saved_model.h5')

def sample(preds, temperature=1.2):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

sentence = "i love the haters because im just gonna " #must be 40 chars long
generated = sentence
for i in range(400):
    x_pred = np.zeros((1, maxlen, total_chars))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.0
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char

print("...Generated: ", generated)
