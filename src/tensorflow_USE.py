
from os.path import isdir
from os import mkdir
import os
import tensorflow as tf
import tensorflow_hub as hub

class TF_USE:
    based_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/'
    def __init__(self, module=None):
        # If folder doesn't exist, create folder
        if not isdir('./tensorflow_USE'): mkdir('./tensorflow_USE') 
        # Point to cache location
        os.environ["TFHUB_CACHE_DIR"] = './tensorflow_USE' 
        
        if tf.__version__.split('.',1)[0]=='2':
            self.embed = hub.KerasLayer(
                self.based_url+'4' if not module else module
            )
        else:
            self.embed = hub.Module(
                self.based_url+'3' if not module else module
            )
    
    def embed_useT(self, documents):
        if tf.__version__.split('.',1)[0]=='2':
            return self.embed(documents)['outputs']
        else:
            with tf.Graph().as_default():
                sentences = tf.compat.v1.placeholder(dtype=tf.string)
                embed = hub.Module(module_url+'3')
                embed_input = embed(sentences)
                session = tf.train.MonitoredSession()
                return session.run(
                    embed_input,
                    {sentences: documents}
                )
