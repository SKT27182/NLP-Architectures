import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import numpy as np


class MemNN:

    def __init__(self,
                 corporus, batch_size=64, max_query_len=16, max_story_len=150, vocab_size=10000, embedding_size=128, k=3):

        self.embedding_size = embedding_size
        self.max_query_len = max_query_len
        self.max_story_len = max_story_len
        self.k = k
        self.batch_size = batch_size

        self.text_tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=None)
        # get the vocab size
        self.text_tokenizer.adapt(corporus)
        self.vocab_size = len(self.text_tokenizer.get_vocabulary())

    def _preprocess_query_and_story(self, query, story):
            
        # query: a list of strings
        # story: a list of strings

        # tokenize the query and story
        tokenized_query = self.text_tokenizer(query)
        tokenized_story = self.text_tokenizer(story)

        # pad the query and story
        padded_query = tf.keras.preprocessing.sequence.pad_sequences(tokenized_query, maxlen=self.max_query_len, padding='post')
        padded_story = tf.keras.preprocessing.sequence.pad_sequences(tokenized_story, maxlen=self.max_story_len, padding='post')

        return padded_query, padded_story
    
    def _preprocess_answer(self, answers):
            
        # answer: a list of strings

        # get the indx of the answer from the vocab
        answers = [self.text_tokenizer.get_vocabulary().index(answer) for answer in answers]

        answers = np.array(answers)
        return answers
                
    
    def embedding(self, inputs, story=False):

        x = inputs
        

        if story:
            x = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True)(x)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True)(x)

        return x
    
    def I(self, inputs):

        # inputs: story, query

        story, query = inputs

        embedded_query = self.embedding(query)

        embedded_story = self.embedding(story, story=True)

        return embedded_story, embedded_query
    
    def G(self, embedded_story):

        # store the story in memory and return the memory
        
        memory = tf.keras.layers.LSTM(embedded_story.shape[-1], input_shape=(None, embedded_story.shape[-1]), return_sequences=True)(embedded_story)
        
        return memory
    
    def O(self, memory, embedded_querry):

        # inputs: memory, embedded_query, k

        # k is the number of hops

        output = embedded_querry

        for i in range(self.k):

            # match the output with the memory, return the index of the memory that matches the output the most

            similarity = tf.keras.layers.dot([output, memory], axes=(2,2))

            # add a lstm layer to make it the dimension of the output
            similarity = tf.keras.layers.LSTM(output.shape[-1], input_shape=(None, similarity.shape[-1]), return_sequences=True)(similarity)

            output = tf.keras.layers.add([output, similarity])

        return output
    
    def R(self, output):
        
        # inputs: output, memory

        answer = tf.keras.layers.LSTM(self.vocab_size)(output)
        answer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(answer)

        return answer
    
    def build(self):
        
        # inputs: story, query

        tokenized_story = tf.keras.Input(shape=[self.max_story_len,], name='story', dtype=np.int32)
        tokenized_query = tf.keras.Input(shape=[self.max_query_len,], name='query', dtype=np.int32)

        embedded_story, embedded_query = self.I([tokenized_story, tokenized_query])

        # do the positional encoding here to the embedded_story and embedded_query

        memory = self.G(embedded_story)

        output = self.O(memory, embedded_query)

        answer = self.R(output)

        model = tf.keras.Model(inputs=[tokenized_story, tokenized_query], outputs=answer)

        self.model = model
    
    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy']):

        self.build()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        
    def fit(self, story, query, answer, epochs, batch_size=4, validation_split=None, **kwargs):

        query, story = self._preprocess_query_and_story(query, story)

        answer = self._preprocess_answer(answer)

        self.model.fit([story, query], answer, epochs=epochs, batch_size=batch_size, validation_split=validation_split, **kwargs)

    def predict(self, story, query):

        query, story = self._preprocess_query_and_story(query, story)

        predictions = self.model.predict([story, query])

        # return the word with the highest probability

        index = np.argmax(predictions, axis=1)

        return [self.text_tokenizer.get_vocabulary()[i] for i in index]