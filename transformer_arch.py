import tensorflow as tf
import numpy as np
import tiktoken

### Tokenizer

class Tokenizer:
  def __init__(self, model_name: str = 'cl100k_base', max_length: int = 128, pad_token_id: int = 0):
    """ Tokenizer warpper - uses default GPT4 tokenizer by default through the OpenAI tiktoken library

    Args:
        model_name: Choose tiktoken model - default is GPT4
        max_length: Choose max truncation length for embedding
        pad_token_id: token ID for padding
    """
    self.tokenizer = tiktoken.get_encoding(model_name)
    self.max_length = max_length
    self.pad_token_id = pad_token_id

  def encode(self, text: str) -> list[int]:
    """ Tokenizes input sequence

    Args:
        text: input string
    Returns:
        List of token IDs
    """
    tokens = self.tokenizer.encode(text)
    return self._pad_or_truncate(tokens)


  def decode(self, tokenids: list[int]) -> str:
    """ Decodes tokenized sequence

    Args:
        tokenids: list of token ids
    Returns:
        decoded string
    """
    return self.tokenizer.decode(tokenids)

  def batch_encode(self, texts: list[str]) -> np.ndarray:
    """ Tokenizes input sequences in batches

    Args:
        texts: List of strings to be encoded
    Returns:
        nd array of token IDs

    """

    token_batches = [self.encode(text) for text in texts]
    return np.array(token_batches)

  def _pad_or_truncate(self, tokens: list[int]) -> list[int]:
    """ Pads or truncates tokenized sequence to match max_length

    Args:
        tokens: list of token ids
    Returns:
        list of token ids of length = max_length

    """
    if len(tokens) > self.max_length:
      return tokens[:self.max_length]
    return tokens + [self.pad_token_id] * (self.max_length - len(tokens))


### Positional Encoder
class PositionalEncoder(tf.keras.layers.Layer):
  def __init__(self, max_length: int, em_dims: int):
    """
    Positional Encoding TF custom layer

    Args:
        max_length: Max sequence length
        em_dims: embeddings dimensions
    Returns:
        None
    """
    super(PositionalEncoder, self).__init__()
    self.pe = self.compute_pe(max_length, em_dims)

  def compute_pe(self, max_length: int, em_dims: int):
    """
    Computues positional encoding using sinusoidal encoding matrix

    Args:
        max_length: max sequence length
        em_dims: embeddings dimensions
    Returns:
        Positional encodings tensor
    """

    positions = np.arange(max_length)[:, None]
    div_const = np.exp(np.arange(0, em_dims, 2) * (-np.log(10000.0) / em_dims))    ### Uses log properties for rearanging ## uses 10,000 from Attention is All You Need

    pos_encodings = np.zeros((max_length, em_dims))
    pos_encodings[:, 0::2] = np.sin(positions * div_const)
    pos_encodings[:, 1::2] = np.cos(positions * div_const)

    return tf.convert_to_tensor(pos_encodings, dtype=tf.float32)
  
  def call(self, inputs):
      """
      Adds positional encodings to input embeddings.
      
      Args:
          inputs: Input tensor of shape (batch_size, seq_length, em_dims)
      Returns:
          Tensor of same shape as inputs with positional encodings added
      """
      
      seq_length = tf.shape(inputs)[1]
      pe_slice = self.pe[:seq_length, :]
      
      return inputs + pe_slice

### Multi-Head Attention Mechanism
class MultiHeadAttn(tf.keras.layers.Layer):
  def __init__(self, em_dims, n_splits):
    """ 
    Computes self-attention of split values at each head in parallel.

    Args:
      em_dims: embeddings dimensions
      n_splits: number of times the embedings should be split for attention computation/number of heads
    Returns:
    """
    super(MultiHeadAttn, self).__init__()
    assert(em_dims % n_splits == 0)
    self.em_dims = em_dims
    self.n_splits = n_splits
    self.q_size = em_dims // n_splits

    self.weights_q = tf.keras.layers.Dense(em_dims)
    self.weights_v = tf.keras.layers.Dense(em_dims)
    self.weights_k = tf.keras.layers.Dense(em_dims)
    self.output_layer = tf.keras.layers.Dense(em_dims)

  def split_heads(self, tensor, batch_size: int):
    """
    Splits embeddings into tensors for each head to compute.
    
    Args:
      tensor: an input tensor of dims (batch_size, seq_length, em_dims)
      batch_size: size of dim 1 of tensor
    Returns:
      tensor of dims (batch_size, num_heads, seq_length, q_size)
    """
    x = tf.reshape(tensor, (batch_size, -1, self.n_splits, self.q_size))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def scaled_dot_product(self, query, value, key, mask=None):
    """
    Computes scaled dot product of the heads and c

    Args:
      query: query matrix (features of interest)
      value: value matrix (embeddings values)
      key: key matrix (mask collections)
      mask: mask to apply if passed in

    Returns:
      Attention scored tensor and attention weights

    """

    numerator = tf.matmul(query, key, transpose_b=True)
    denom = tf.sqrt(tf.cast(self.q_size, tf.float32))


    func_term = numerator/denom
    if mask != None:
      func_term += (mask* -1e9)

    attn_weights = tf.nn.softmax(func_term)

    return attn_weights, tf.matmul(attn_weights, value)

  def call(self, query, value, key, mask=None):
    """
    Executes a forward call of the multihead attention mechanism

    Args:
      query: query matrix (features of interest)
      value: vlaue matrix (embeddings values)
      key: key matrix (mask collections)
      mask: mask to apply if passed in

    Returns:
      matrix of size batch x sequence length x embedding dimensions with contextualized embeddings
    """
    q = self.weights_q(query)
    v = self.weights_v(value)
    k = self.weights_k(key)

    batch_size = tf.shape(q)[0]

    split_q = self.split_heads(q, batch_size)
    split_v = self.split_heads(v, batch_size)
    split_k = self.split_heads(k, batch_size)

    attn_weights, attn_mtx = self.scaled_dot_product(split_q, split_v, split_k, mask)

    attn_mtx = tf.transpose(attn_mtx, perm=[0, 2, 1, 3])
    attn_mtx = tf.reshape(attn_mtx, (batch_size, -1, self.em_dims))

    return self.output_layer(attn_mtx)
  

### Feed Forward Network Layer
class FFN(tf.keras.layers.Layer):
  def __init__(self, em_dims: int, ff_dims: int):
    """
    Creates feedforward network layer

    Args:
      em_dims: embedding dimensions
      ff_dims: dimensions for the dense layers
    
    Returns:
    """
    super(FFN, self).__init__()
    self.em_dims = em_dims
    self.ff_dims = ff_dims

    self.dense = tf.keras.layers.Dense(ff_dims)  #Expands model dimensionality
    self.activation = tf.keras.layers.ReLU()    
    self.dense_rev = tf.keras.layers.Dense(em_dims)    #Shrinks back to tensor dimensionality

  def call(self, tensor):
    """
    Executes a forward call of the FFN layer

    Args:
      tensor: input tensor for network call

    Returns:
      normalized tensor of input dimensions 
    """

    expanded = self.dense(tensor)
    expanded = self.activation(expanded)
    return self.dense_rev(expanded)
  


### Seperate Add+Norm layer rather than built into FFN or MH_ATTN
class Add_Norm(tf.keras.layers.Layer):
  def __init__(self):
    """
    Skip conneciton layer. Adds input to normalized multi-headed attention outputs of FNN outputs

    Args:
      em_dims: embeddings dimensions
    
    Returns:
    """
    super(Add_Norm, self).__init__()
    self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, input_tensor, operated_tensor):
    """
    Executres call for add and norm layer.

    Args:
      input_tensor: embeddings tensor before FFN or MH_ATTN computation
      operated_tensor: embeddings tensor that has been passed though the FFN or MH_ATTN

    Returns:
      smoothed embeddings tensor
    """  

    return self.norm_layer(operated_tensor + input_tensor)
 

  






    