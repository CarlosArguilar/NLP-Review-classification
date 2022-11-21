import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

from parameters import params


class Model:

    def __init__(self, df: pd.DataFrame, x: str, y: str) -> None:
        '''
        df: Dataframe with the data
        x: Column of the dataframe that contains the text
        y: Columns of categories that will be predicted
        '''

        # Store params as instance variables
        self.df = df
        self.x = x
        self.y = y

        self.params = params

        # Factorize y column
        self.df['y'] = pd.factorize(self.df[self.y])[0]

        # Getting dict to relate int from factorization with str category
        aux_df = self.df[['y', self.y]].drop_duplicates(self.y)
        aux_zip = zip(aux_df['y'], aux_df[self.y])
        self.dict_categories = {factor: text for factor, text in aux_zip}

    def tokenization(self):
        self.tokenizer = Tokenizer(num_words=self.params['model']['max_tokens'], oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.df[self.x])

        training_sequences = self.tokenizer.texts_to_sequences(self.df[self.x])
        self.padded_training_seq = pad_sequences(
            training_sequences, padding='post', truncating='post')

        self.max_length = len(self.padded_training_seq[0])

    def train(self, model: keras.Sequential) -> keras.callbacks.History:

        self.model = model

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        self.model.summary()

        history = self.model.fit(
            self.padded_training_seq, self.df['y'],
            epochs=self.params['model']['epochs'], verbose=1,
            validation_split=self.params['model']['validation_split']
        )

        return history

    def save(self, name: str) -> None:

        # Save model
        self.model.save(os.path.join('saved', 'model.h5'), save_format='h5')
        
        # Save tokenizer
        with open(os.path.join('saved', 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        # Save dictionary of categories
        with open(os.path.join('saved', 'categories.pkl'), 'wb') as f:
            pickle.dump(self.dict_categories, f)