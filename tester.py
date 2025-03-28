import transformer_arch as arch
import numpy as np
import tensorflow as tf
import blocks as bk


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
em_dims = 512
num_heads = 8

sample_msa = arch.MultiHeadAttn(em_dims, num_heads)

x = tf.random.uniform((batch_size, seq_length, em_dims))

output = sample_msa(x, x, x)
print("Output shape:", output.shape)




### FFN Tester:
em_dims = 512
ff_dims = 2048
batch_size = 2
seq_length = 10

ffn_layer = arch.FFN(em_dims, ff_dims)
test_tensor = tf.random.uniform((batch_size, seq_length, em_dims))

output_tensor = ffn_layer(test_tensor)
print("Output shape:", output_tensor.shape)



### Add & Norm Tester:
batch_size, seq_length, em_dims = 2, 10, 512
input_tensor = tf.random.uniform((batch_size, seq_length, em_dims))
operated_tensor = tf.random.uniform((batch_size, seq_length, em_dims))

add_norm_layer = arch.Add_Norm()
output_tensor = add_norm_layer(input_tensor, operated_tensor)

print("Output shape:", output_tensor.shape)



### Encoder Block Tester:
batch_size = 2
seq_length = 10
em_dims = 512
n_heads = 8

encoder_block = bk.EncoderBlock(em_dims, n_heads)
test_input = tf.random.uniform((batch_size, seq_length, em_dims))

output = encoder_block(test_input)
print("Encoder block output shape:", output.shape)


### Decoder Block Tester:
batch_size = 2
seq_length = 10
em_dims = 512
n_heads = 8

decoder_block = bk.DecoderBlock(em_dims, n_heads)

decoder_input = tf.random.uniform((batch_size, seq_length, em_dims))
encoder_input = tf.random.uniform((batch_size, seq_length, em_dims))
mask = tf.ones((batch_size, 1, seq_length, seq_length))  # Example mask

output = decoder_block(decoder_input, encoder_input, mask)
print("Decoder block output shape:", output.shape)