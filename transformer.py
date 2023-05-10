# positional encoding
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer,Embedding,Dense,LayerNormalization,Dropout,Flatten
from tensorflow.keras.initializers import GlorotNormal

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncoding(Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()

        self.embed_size = embed_size
        self.embedding = Embedding(input_dim=vocab_size,output_dim=embed_size)
        self.pos_encoding = tf.constant(positional_encoding(length=window_size,depth=embed_size),dtype=tf.float32)

    def call(self, x):
        # scaler
        emb_sz = float(self.embed_size)
        scaler = np.sqrt(emb_sz)
        out = self.embedding(tf.cast(x,tf.float32)) * scaler
        out = tf.add(out,self.pos_encoding)
        return out

class AttentionHead(Layer):

    def __init__(self,input_size,output_size):
        super(AttentionHead,self).__init__()

        self.W_q = tf.Variable(GlorotNormal(seed=22)(shape=[input_size,output_size]))
        self.W_k = tf.Variable(GlorotNormal(seed=22)(shape=[input_size,output_size]))
        self.W_v = tf.Variable(GlorotNormal(seed=22)(shape=[input_size,output_size]))

    def call(self,inputs):

        # perform dot products
        Q = tf.tensordot(inputs,self.W_q,axes=1)
        K = tf.tensordot(inputs,self.W_k,axes=1)
        V = tf.tensordot(inputs,self.W_v,axes=1)

        # perform matrix multiplication between query and key 
        out = tf.matmul(Q,K,transpose_b=True)
        
        # normalizing
        key_size = K.get_shape()[1]
        out = out / tf.sqrt(float(key_size))

        # performing softmax
        out = tf.nn.softmax(out)

        # atten_matrix matmul aginst the values
        out = tf.matmul(out,V)

        return out
    
class MultiHeadAttention(Layer):

    def __init__(self,emb_sz,num_of_heads):
        super(MultiHeadAttention,self).__init__()
        
        self.num_of_heads = num_of_heads
        self.head_size = emb_sz // num_of_heads 
        self.atten_heads = [AttentionHead(self.head_size,self.head_size) for _ in range(num_of_heads)]
    
    def call(self,inputs):

        # split tensors based of number of heads 
        input_splits = tf.split(inputs,num_or_size_splits=self.num_of_heads,axis=2)

        # output from each attention head
        out = [self.atten_heads[i](input_splits[i]) for i in range(self.num_of_heads)]
        
        # concatenating the results
        result = out[0]
        for i in range(1,len(out)):
            result = tf.concat([result,out[i]],axis=2)

        return result


class FeedForward(Layer):

    def __init__(self,emb_sz):
        super(FeedForward,self).__init__()

        self.dense = Dense(units=emb_sz,activation='leaky_relu')
    
    def call(self,inputs):
        
        out = self.dense(inputs)
        
        return out

class TransformerBlock(Layer):
    
    def __init__(self,emb_sz,num_of_heads):
        super(TransformerBlock,self).__init__()

        self.mh_att = MultiHeadAttention(emb_sz,num_of_heads)
        self.ff = FeedForward(emb_sz)
        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()
        self.dropout = Dropout(rate=0.25)

    def call(self,inputs):

        out = self.mh_att(inputs)
        out = tf.add(out,inputs)
        out = self.norm_1(out)
        out = self.dropout(out)
        ff_out = self.ff(out)
        ff_out = tf.add(ff_out,out)
        ff_out = self.norm_2(ff_out)
        ff_out = self.dropout(ff_out)

        return ff_out

class Cutter(Layer):

    def __init__(self):
        super(Cutter,self).__init__()
    
    def call(self):
        return


class TransformerEncoder(Model):
    
    def __init__(self,vocab_size,emb_sz=64,num_of_blocks=1,num_of_atten_heads=4,window_size=1000):
        super(TransformerEncoder,self).__init__()
        
        self.num_of_blocks = num_of_blocks
        self.vocab_size = vocab_size
        self.embed = PositionalEncoding(vocab_size,emb_sz,window_size) # window size is the max length of the sequence
        self.transformer_block = TransformerBlock(emb_sz,num_of_atten_heads)
        self.dense = Dense(units=emb_sz,activation='leaky_relu')
        self.classifier = Dense(units=1,activation='sigmoid')
        self.flatten = Flatten()
    
    def call(self,inputs):
        

        # Embedding with Position
        out = self.embed(inputs)

        # Module 1: Transformer Blocks
        for _ in range(self.num_of_blocks):
            out = self.transformer_block(out)

        # Linear Layer and Classifier
        out = self.flatten(out)
        out = self.dense(out)
        out = self.classifier(out)

        # Module 2: BedGraph Neural Network
        # out_bedgraph = None
        # result = tf.concat([out,out_bedgraph],axis=-1)

        return out
