import transformer_arch as arch
import numpy as np
import tensorflow as tf


##Tokenizer Test
tokenizer = arch.Tokenizer(max_length=5)

text = "This is a string for testing my tokenizer class."
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print("Tokens:", tokens)
print("Decoded:", decoded)

batch_texts = ["Hello world!", "I am building an llm."]
batch_tokens = tokenizer.batch_encode(batch_texts)
print("Batch Tokenized:\n", batch_tokens)


###Pos Encoder Test

max_length = 100
em_dims = 512
post_enc_layer = arch.PositionalEncoder(max_length, em_dims)

samp_input = tf.zeros((1,max_length, em_dims))
encoded_in = post_enc_layer(samp_input)

print('Input Shape:', encoded_in.shape)


### Self Attn Test
batch_size = 2
seq_length = 5
d_model = 512
num_heads = 8

sample_msa = arch.MultiHeadSelfAttention(d_model, num_heads)

x = tf.random.uniform((batch_size, seq_length, d_model))

output = sample_msa(x, x, x)
print("Output shape:", output.shape)
