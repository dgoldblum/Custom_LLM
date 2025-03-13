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
class PositionalEncoder(tf.keras.layers.layers):
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

    positions = np.arrange(max_length)[:, None]
    div_const = np.exp(np.arange(0, em_dims, 2) * (-np.log(10000.0) / em_dims))    ### Uses log properties for rearanging ## uses 10,000 from Attention is All You Need

    pos_encodings = np.zeros((max_length, em_dims))
    pos_encodings[:, 0::2] = np.sin(positions * div_const)
    pos_encodings[:, 1::2] = np.cos(positions * div_const)

    return tf.convert_to_tensor(pos_encodings, dtype=tf.float32)


### Multi-Head Attention Mechanism
class MultiHeadAttn:
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

    weights_q = tf.layers.Linear(em_dims)
    weights_v = tf.layers.Linear(em_dims)
    weights_k = tf.layers.Linear(em_dims)

    output = tf.layers.Linear(em_dims)

    def split_heads(self, tensor, batch_size):
      """
      Splits embeddings into tensors for each head to compute.
      
      Args:
        tensor: an input tensor of dims (batch_size, seq_length, em_dims)
        batch_size: size of dim 1 of tensor
      Returns:
        tensor of dims (batch_size, num_heads, seq_length, q_size)
      """

      x = tf.reshape(x, (batch_size, -1, self.n_splts, self.q_size))
      #Splits embeddings dims into heads x query_size
      return tf.transpose(x, perm = [0, 2, 1 ,3]) #Transposes since we want number of heads earlier in matrix for computation
  
  def scaled_dot_product(self, query, value, key):
    """
    Computes scaled dot product of the heads and c
    
    """

    