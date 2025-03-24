import tensorflow as tf
import transformer_arch as arch



class EncoderBlock():
    def __init__(self, input_dims, n_splits):  #Might need to change to em_dims
        """
        The encoding block of the transformer. Uses components built in transformer_arch.py

        Args:
            input_dims: dimensions of embeddings tensor
            n_splits: number of splits for batching in multi headed attn 
        Returns:
        """
        super(EncoderBlock, self).__init__()

        self.input_dims = input_dims
        self.n_splits = n_splits

        self.mh_attn = arch.MultiHeadAttn(input_dims, n_splits)
        self.ffn = arch.FFN(input_dims, 2048)  #arbitrarily chosen expansion dim, double check
        self.addnorm = arch.Add_Norm(input_dims)
    
    def encoder_call(self, input):
        """
        Runs encoder block:

        Args:
            input: embeddings matrix input

        Returns:
            embeddings encoder output (This should probably be changed)
        """
        ### I don't know if I actually need to forward call or not
        attn_filtered = self.mh_attn.forward_call(input, input, input) #Fix args
        normed_attn = self.addnorm.forward_call(input, attn_filtered)

        ffn_pass = self.ffn.forward_call(normed_attn)
        return self.addnorm.forward_call(normed_attn, ffn_pass)


class DecoderBlock():
    def __init__(self, o_input_dims, n_splits, mask):
        """
        The decoder block of the transformer

        Args:
            o_input_dims: output embeddings tensor dimensions
            n_splits: number of splits for batching in multi-headed attn

        Returns:
        """
        super(DecoderBlock, self).__init__()

        self.o_input_dims = o_input_dims
        self.n_splits = n_splits

        #Not sure if this should be an init input or not
        self.mask = mask
        
        self.mh_attn = arch.MultiHeadAttn(o_input_dims, n_splits)
        self.ffn = arch.FFN(o_input_dims, 2048) #Same arbitrary dims as encoder
        self.addnorm = arch.Add_Norm(o_input_dims)
    
    def decoder_call(self, o_input, enocder_input, mask):
        """
        Calls decoder block of the transformer

        Args:
            o_input: output embeddings tensor
            encoder_input: output from encoding as an input for cross attention
        
        Returns:
            tensor for outputing next word/embedding probabilites
        """

        masked_attn = self.mh_attn.forward_call(o_input, o_input, o_input, mask=mask)
        normed_masked = self.addnorm.forward_call(o_input, masked_attn)

        cross_attn = self.mh_attn.forward_call(normed_masked, enocder_input, enocder_input)
        normed_cross = self.addnorm.forward_call(normed_masked, cross_attn)

        ffn_pass = self.ffn.forward_call(normed_cross)
        return self.addnorm.forward_call(normed_cross, ffn_pass)


