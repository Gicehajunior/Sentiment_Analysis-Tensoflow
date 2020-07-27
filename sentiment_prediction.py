import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import os
import data_manipulation
import sentiment_model


print("Enter a sample text to test")
text_input = input()

next_words = 30

for _ in range(next_words):
    # do some tokenization
    tokenizer = Tokenizer()
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    
    # do some padding
    token_list = pad_sequences([token_list], maxlen=data_manipulation.padded_sequences[2]-1, padding='pre')
    
    # predicted classes
    predicted = sentiment_model.model.predict_classes(token_list, verbose=0)
    
    
    output_word = ""
    corpus = tokenizer.word_index.items()
    for word, index in corpus:
        if index == predicted:
            output_word = word
            
            break
        text_input += " " + output_word
print(text_input)





