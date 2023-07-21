from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# input text
texts = ["Support among scientists for human-caused climate change is below 97%"]

# word embeddings preprocessing parameters
max_len = 500
top_words = 5000
max_words = 10000
embedding_dim = 300

# load model 
model = load_model('Models/Full_BiLSTM.h5')

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
seqs = tokenizer.texts_to_sequences(texts)
data = pad_sequences(seqs, maxlen=max_len)

print(model.predict(data))