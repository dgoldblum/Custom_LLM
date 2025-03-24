import tensorflow as tf
import blocks as bk
import transformer_arch as arch

class Transformer():
    def __init__(self, inputs, outputs):
        """
        Building the transformer from individual architecture and blocks

        Args:

        Returns:

        """

        super(Transformer, self).__init__()

        self.enc_block = bk.EncoderBlock()
        self.dec_block = bk.DecoderBlock()

        self.input_dims = tf.shape(inputs)
        self.output_dims = tf.shape(outputs)

        self.in_pos_enc = arch.PositionalEncoder(128, self.input_dims)
        self.out_pos_enc = arch.PositionalEncoder(128, self.output_dims)

        self.linear_layer = tf.layers.Linear(self.output_dims)
        self.softmax = tf.nn.softmax()



