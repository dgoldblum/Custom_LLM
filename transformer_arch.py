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
