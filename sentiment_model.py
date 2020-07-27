import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import os
import data_manipulation

models_filepath_to_save = "app-resources/models-folder"

def create_model():
    model = keras.Sequential()
    model.add(layers.Embedding(total_words, 240, input_length=padded_sequences[2]-1))
    model.add(layers.Bidirectional(layers.LSTM(150)))
    model.add(layers.Dense(total_words, activation='softmax'))
    
    return model

def model_history(model):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    #compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #create a fit
    history = model.fit(padded_sequences[0], padded_sequences[1], epochs=100, verbose=1)
    return [history, model]

def save_model(model_path, model):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    else:
        model.save(model_path)
    

create_model = create_model()

#display the model archtecture
create_model.summary()

history, model = model_history(create_model)
#print(history)
#print(model)

#saving the model
save_model(models_filepath_to_save, model)
