import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import os


file_path = "app-resources/files/Poems.txt"


def read_file(file_path):
    corpus = []
    with open(file_path,"r") as file:
        for line in file.readlines():
            line = line.lower().split('\n')
            
            corpus.append(line)
            
        return corpus
        
        
#tokenization of the corpus
def tokenize_texts(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    
    return tokenizer

def create_sequences(corpus, tokenizer, total_words):
    input_sequences = []
    # Remove any empty lines
    corpus = [line for line in corpus if line != '']
    for line in corpus:
        #create token list
        token_list = tokenizer.texts_to_sequences([line])[0]
        
        for i in range(1, len(token_list)):
            #create an Ngram sequence
            ngram_sequences = token_list[:i+1]

            input_sequences.append(ngram_sequences)

            maximum_sequence_len = max([len(input_sequence) for input_sequence in input_sequences])
            padded_sequences = numpy.array(pad_sequences(input_sequences, maxlen=maximum_sequence_len, padding='pre'))
            input_sequences = padded_sequences[:,:-1]
            labels = padded_sequences[:,-1]
            hot_encoded_labels = tensorflow.keras.utils.to_categorical(labels, num_classes=total_words)
            
            return [input_sequences, hot_encoded_labels, maximum_sequence_len]

corpus = read_file(file_path)

tokenizer = tokenize_texts(corpus)

tokenized_corpus = tokenizer.word_index

total_words = len(tokenized_corpus)+1
print(tokenized_corpus)

print(total_words)

padded_sequences = create_sequences(corpus, tokenizer, total_words)                                                
#print(padded_sequences[0])
#print(padded_sequences[1])
