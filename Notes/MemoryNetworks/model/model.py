import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import numpy as np
import tensorflow_hub as hub

hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128],
                           input_shape=[], dtype=tf.string)


class MemNN:

    def __init__(self,
                 corporus, max_query_len=1, max_story_len=10, vocab_size=10000, embedding_size=128, k=3):
        
        """
        corporus: a list of strings that contains all the words in the corporus
        
        max_query_len: the max length of the query, default is 1, which means the query is a single sentence
        
        max_story_len: the max length of the story, default is 2, which means the story is a list of 2 sentences
        
        vocab_size: the size of the vocabulary, default is 10000
        
        embedding_size: the size of the embedding, default is 128
        
        k: the number of hops, default is 3
        
        """

        self.embedding_size = embedding_size
        self.max_query_len = max_query_len
        self.max_story_len = max_story_len
        self.k = k
        self.batch_size = 8

        self.text_tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=max_query_len)
        # get the vocab size
        self.text_tokenizer.adapt(corporus)
        self.vocab_size = len(self.text_tokenizer.get_vocabulary())

    def _pad_stories(self, stories):
        batch_size, n_sentence = stories.shape

        padding = np.array([[""] * (self.max_story_len - n_sentence)] * batch_size)
        stories = np.concatenate((stories, padding), axis=1)

        return stories

    def embedding(self, inputs):
            
            """
            inputs: a list of strings
            
            return: a list of embeddings
            
            """
            
            outputs = []

            x = inputs

            for _, single_input in enumerate(x):


                single_input = hub_layer(single_input)

                outputs.append(single_input)

            outputs = tf.stack(outputs)

            return outputs
    
    def _preprocess_answer(self, answers):
            
        # answer: a list of strings

        # get the indx of the answer from the vocab
        answers = [self.text_tokenizer.get_vocabulary().index(answer) for answer in answers]

        answers = np.array(answers, dtype=np.float32).reshape(-1, 1)
        return answers
                
    
    def I(self, inputs):

        # inputs: story, query

        story, query = inputs

        embedded_query = self.embedding(query)

        embedded_story = self.embedding(story)

        return embedded_story, embedded_query
    
    def G(self, embedded_story):

        # store the story in memory and return the memory
        
        memory = tf.keras.layers.LSTM(self.embedding_size, input_shape=(None, self.embedding_size), return_sequences=True)(embedded_story)
        
        return memory
    
    def O(self, memory, embedded_querry):

        """
        
        memory: the memory that stores the story

        embedded_querry: the embedded query

        return: the output of the model, which will be further processed by the R function
        """
        output = embedded_querry

        memory = tf.nn.l2_normalize(memory, axis=2)

        for _ in range(self.k):

            output = tf.nn.l2_normalize(output, axis=2)

            # match the output with the memory, return the index of the memory that matches the output the most
            similarity =  tf.matmul(memory, output, transpose_b=True)

            # get the memory that matches the output the most
            idx = tf.argmax(similarity, axis=1)

            batch_indices = tf.range(tf.shape(memory)[0], dtype=idx.dtype)
            indices = tf.stack([batch_indices, tf.squeeze(idx, axis=1)], axis=1)
            most_similar_memory = tf.gather_nd(memory, indices)

            # add the most_similar_memory to the output
            output = tf.keras.layers.add([output, most_similar_memory])

        return output
    
    def R(self, output):
        
        """

        output: the output of the model, which will be further processed by the R function

        return: the answer of the model

        """

        answer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.embedding_size, return_sequences=True))(output)
        answer = tf.keras.layers.Dense(self.vocab_size, activation='relu')(answer)
        answer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(answer)

        return answer
    
    def _build(self):
        
        # inputs: story, query

        # tokenized_story = tf.keras.Input(batch_size=self.batch_size, shape=[self.max_story_len], name='story', dtype=tf.string)
        # tokenized_query = tf.keras.Input(batch_size=self.batch_size, shape=[self.max_query_len], name='query', dtype=tf.string)

        # embedded_story, embedded_query = self.I([tokenized_story, tokenized_query])

        embedded_story = tf.keras.Input(batch_size=self.batch_size, shape=[self.max_story_len, self.embedding_size], name='story', dtype=tf.float32)
        embedded_query = tf.keras.Input(batch_size=self.batch_size, shape=[self.max_query_len, self.embedding_size], name='query', dtype=tf.float32)
        # do the positional encoding here to the embedded_story and embedded_query

        memory = self.G(embedded_story)

        output = self.O(memory, embedded_query)

        answer = self.R(output)

        model = tf.keras.models.Model(inputs=[embedded_story, embedded_query], outputs=answer)

        self.model = model
    
    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy']):

        self._build()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        
    def fit(self, story, query, answer, epochs, batch_size=32, validation_split=0.2, **kwargs):

        self.batch_size = batch_size

        answer = self._preprocess_answer(answer)

        # check if the story length is less than the max_story_len, then pass " " to the story

        if story.shape[1] < self.max_story_len:
            story = self._pad_stories(story)

        story, query = self.I([story, query])

        self.model.fit([story, query], answer, epochs=epochs, validation_split=validation_split, **kwargs)

    def predict(self, story, query):

        self.batch_size = story.shape[0]

        if story.shape[1] < self.max_story_len:
            story = self._pad_stories(story)

        story, query = self.I([story, query])

        predictions = self.model.predict([story, query])

        # return the word with the highest probability

        index = np.argmax(predictions, axis=1)

        return [self.text_tokenizer.get_vocabulary()[i] for i in index]