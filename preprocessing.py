import logging
import logging.handlers

import spacy
from spellchecker import SpellChecker
from nltk.corpus import stopwords as nltk_stopwords

import time
import pandas as pd
from tqdm import tqdm

from typing import Callable


class PreProcessingMethods:
    '''
    Class containing methods of preprocessing.

    The class PreProcessor must inherit this class.
    '''

    ########################################################################
    # Lower case

    def convert_to_lower_case(self) -> None:
        '''
        Method to convert texts to lower case
        '''
        self.text_series = self.text_series.str.lower()

    ########################################################################
    # Removing numbers

    def remove_numbers(self) -> None:
        '''
        Step to remove number
        '''

        self.text_series = self.text_series.str.replace('\d+', '', regex=True)

    ########################################################################
    # Removing unwanted characters

    def remove_non_word_characters(self) -> None:
        '''
        Removes characters that are not word characters (regex)
        '''

        self.text_series = self.text_series.str.replace(
            '[^\w\s]', '', regex=True)

    ########################################################################
    # Removing URL's

    def remove_urls(self) -> None:
        '''
        Remove urls by regex substitution
        '''
        regex_pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
        self.text_series = self.text_series.str.replace(
            regex_pattern, '', regex=True)

    ########################################################################
    # Spell correction

    def __correct_spell(self, text: str) -> str:
        '''
        Auxiliary function that perform spell correction in text
        '''
        nlp_text = self.nlp(text)

        checked_texts = []
        for token in nlp_text:
            correction_return = self.spell.correction(token.text)
            checked_text = correction_return if correction_return else token.text

            checked_texts.append(checked_text)

        return ' '.join(checked_texts)

    def correct_spell(self) -> None:
        '''
        Apply spell correction (word by word)
        '''
        
        self.text_series = self.text_series.progress_apply(self.__correct_spell)

    ########################################################################
    # Lemmatization

    def __lemmatize(self, text: str) -> str:
        '''
        Auxiliary function that perform lemmatization in text
        '''
        nlp_text = self.nlp(text)

        lemmatized_tokens = [token.lemma_ for token in nlp_text]

        return ' '.join(lemmatized_tokens)

    def lemmatization(self) -> pd.Series:

        self.text_series = self.text_series.progress_apply(self.__lemmatize)

        return self.text_series

    ########################################################################
    # Stopwords removal

    def __stopword_removal(self, text: str) -> str:
        nlp_text = self.nlp(text)

        new_text = [
            token.text for token in nlp_text if token.text not in self.stopwords
            ]

        return ' '.join(new_text)

    def remove_stopwords(self) -> pd.Series:
        '''
        Remove stop words
        '''

        self.text_series = self.text_series.progress_apply(self.__stopword_removal)

        return self.text_series

    ########################################################################
    # Truncate text

    def truncate_text(self, max_length: int = 100) -> None:
        '''
        Truncate texts to max length
        '''

        self.text_series = self.text_series.str[:max_length]


class PreProcessor(PreProcessingMethods):
    '''
    Class to execute preprocessing pipeline.

    All methods (names) that correspont to some step of the pipeline
    must start with 'step_', to be executed in the run_pipeline method.

    As consequence, no method that does not correspond to a preprocessing
    step should have a name starting with 'step_'.
    '''

    class_attributes_loaded = False

    @classmethod
    def load_class_attributes(cls) -> None:
        '''
        Class to load attributes that will be used in all instances,
        in order to load only one time and be shared across all instances.
        '''
        # Load spacy model
        cls.nlp = spacy.load('en_core_web_md')

        # Initiate spell checker
        cls.spell = SpellChecker(language='en')

        # Get stopwords
        cls.stopwords = nltk_stopwords.words('english')

        cls.class_attributes_loaded = True

    def __init__(self, text: pd.Series or str) -> None:

        # Load class attributes if not loaded
        if not PreProcessor.class_attributes_loaded:
            PreProcessor.load_class_attributes()

        # identify the preprocessing methods of the PreProcessingMethods class
        self.steps = self.identify_preprocessing_methods()

        # Convert input to series if it's text
        if isinstance(text, str):
            text = pd.Series([text])
        else:
            assert isinstance(text, pd.Series)

        # Assign parameters to instance variables
        self.text_series = text

        # Activate tqdm module to print progess
        tqdm.pandas()

        # Configuration of logger
        if not hasattr(PreProcessor, 'logger'):
            self.configure_logger()

    def identify_preprocessing_methods(self) -> list:
        '''
        Return a list containing the methods that can be used in the pipe line
        '''

        preprocessing_methods = []

        for attr_name in PreProcessingMethods.__dict__:
            attr = getattr(self, attr_name)

            # Skip non callable methods
            if not isinstance(attr, Callable):
                continue

            # Skip private and dunder methods
            if attr.__name__[:2] == '__':
                continue

            preprocessing_methods.append(attr)

        return preprocessing_methods

    def configure_logger(self) -> None:
        
        formatter = logging.Formatter(
            '%(asctime)s:\t%(message)s',
            datefmt='%d/%m/%Y %H:%M:%S'
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        PreProcessor.logger = logging.getLogger()
        PreProcessor.logger.setLevel(logging.INFO)
        PreProcessor.logger.addHandler(stream_handler)

    def list_available_steps(self) -> list:
        '''
        List all available preprocessing steps.
        '''
        steps = [attr.__name__ for attr in self.steps]

        return steps

    def run_pipeline(self, steps: tuple) -> pd.Series:
        '''
        Function to execute pipeline of pre-processing.
        Steps will be executed in the same order that they appear in the tuple
        '''

        steps_methods = [getattr(self, step) for step in steps]

        for step_method in steps_methods:
            print('Running', step_method.__name__)
            # run step method
            step_method()

        return self.text_series
