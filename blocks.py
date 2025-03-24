import tensorflow as tf
import transformer_arch as arch

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, input_dims, n_splits):
        """
        The encoding block of the transformer. Uses components built in transformer_arch.py

        Args:
            input_dims: dimensions of embeddings tensor
            n_splits: number of attention heads 
        """
        super(EncoderBlock, self).__init__()

        self.input_dims = input_dims
        self.n_splits = n_splits

        # Multi-Head Attention layer
        self.mh_attn = arch.MultiHeadAttn(input_dims, n_splits)

        # Feed Forward Network layer
        self.ffn = arch.FFN(input_dims, 2048)  # Expansion dimension (typically 4x embeddings)

        # Add & Norm layers (twice — for attention and FFN outputs)
        self.addnorm1 = arch.Add_Norm()
        self.addnorm2 = arch.Add_Norm()

    def call(self, inputs):
        """
        Runs encoder block:

        Args:
            inputs: Embeddings matrix input (batch_size, seq_len, input_dims)

        Returns:
            Encoder output tensor (batch_size, seq_len, input_dims)
        """
        # Multi-head self-attention with skip connection and normalization
        attn_output = self.mh_attn(inputs, inputs, inputs)
        normed_attn = self.addnorm1(inputs, attn_output)

        # Feedforward network with skip connection and normalization
        ffn_output = self.ffn(normed_attn)
        output = self.addnorm2(normed_attn, ffn_output)

        return output


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, input_dims, n_splits):
        """
        The decoder block of the transformer.

        Args:
            input_dims: Dimensions of the decoder's input embeddings tensor.
            n_splits: Number of attention heads (multi-head attention).
        """
        super(DecoderBlock, self).__init__()

        self.input_dims = input_dims
        self.n_splits = n_splits

        # Separate attention layers for masked and cross-attention
        self.masked_attn = arch.MultiHeadAttn(input_dims, n_splits)
        self.cross_attn = arch.MultiHeadAttn(input_dims, n_splits)

        # Feedforward network layer
        self.ffn = arch.FFN(input_dims, 2048)  # Same expansion dims as encoder

        # Add & Norm layers (3 times — after each major step)
        self.addnorm1 = arch.Add_Norm()
        self.addnorm2 = arch.Add_Norm()
        self.addnorm3 = arch.Add_Norm()

    def call(self, decoder_input, encoder_input, mask):
        """
        Calls the decoder block of the transformer.

        Args:
            decoder_input: Decoder's input embeddings tensor (batch_size, seq_len, input_dims).
            encoder_input: Output from the encoder, used for cross-attention.
            mask: Mask tensor to prevent attending to future tokens (for masked self-attention).

        Returns:
            Tensor representing the next token/embedding probabilities.
        """

        # Masked self-attention (decoder attends to its past tokens only)
        masked_attn_output = self.masked_attn(decoder_input, decoder_input, decoder_input, mask=mask)
        normed_masked = self.addnorm1(decoder_input, masked_attn_output)

        # Cross-attention (decoder attends to encoder's output)
        cross_attn_output = self.cross_attn(normed_masked, encoder_input, encoder_input)
        normed_cross = self.addnorm2(normed_masked, cross_attn_output)

        # Feedforward network with skip connection and normalization
        ffn_output = self.ffn(normed_cross)
        output = self.addnorm3(normed_cross, ffn_output)

        return output


