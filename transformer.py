import tensorflow as tf
import blocks as bk
import transformer_arch as arch
import time

class Transformer(tf.keras.Model):
    def __init__(self, input_dims, output_dims, n_enc, n_dec, d_model=128, n_heads=8, ff_dim=2048, tie_weights=False):
        """
        Initializes the Transformer model with encoder and decoder stacks.
        Args:
            input_dims: Input embedding dimensions.
            output_dims: Output embedding dimensions.
            n_enc: Number of encoder blocks.
            n_dec: Number of decoder blocks.
            d_model: Embedding dimension size (default 128).
            n_heads: Number of attention heads (default 8).
            ff_dim: Dimension of the feed-forward network (default 2048).
            tie_weights: Whether to tie encoder embedding weights to final output projection.
        """
        super(Transformer, self).__init__()

        self.n_enc = n_enc
        self.n_dec = n_dec
        self.tie_weights = tie_weights

        # Positional encoders
        self.in_pos_enc = arch.PositionalEncoder(d_model, input_dims)
        self.out_pos_enc = arch.PositionalEncoder(d_model, output_dims)

        # Stacking encoder and decoder blocks
        self.encoder_blocks = [bk.EncoderBlock(d_model, n_heads) for _ in range(n_enc)]
        self.decoder_blocks = [bk.DecoderBlock(d_model, n_heads) for _ in range(n_dec)]

        # Final output projection
        self.final_dense = tf.keras.layers.Dense(output_dims)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, outputs, mask=None):
        """
        Runs the full Transformer model.
        Args:
            inputs: Input tensor for encoder.
            outputs: Target tensor for decoder.
            mask: Optional masking for the decoder (prevents looking ahead).
        Returns:
            Final output tensor after encoding/decoding.
        """
        # Encode inputs
        enc_outputs = self.in_pos_enc(inputs)
        for block in self.encoder_blocks:
            enc_outputs = block(enc_outputs)

        # Decode outputs (target embeddings)
        dec_outputs = self.out_pos_enc(outputs)
        for block in self.decoder_blocks:
            dec_outputs = block(dec_outputs, enc_outputs, mask)

        # Final output projection and softmax
        final_output = self.final_dense(dec_outputs)
        return self.softmax(final_output)

    def create_causal_mask(self, seq_len):
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask  # Upper triangular mask

    # def transformer_lr_schedule(self, d_model, warmup_steps=4000):
    #     def lr(step):
    #         arg1 = step ** -0.5
    #         arg2 = step * (warmup_steps ** -1.5)
    #         return (d_model ** -0.5) * min(arg1, arg2)
    #     return tf.keras.optimizers.schedules.LearningRateSchedule(lr)
    
    def build(self, input_shape):
        """
        Ensures weight tying happens after model weights are initialized.
        """
        super().build(input_shape)
        if self.tie_weights:
            self.final_dense.kernel = self.in_pos_enc.embedding.embeddings

    def create_model(self, input_shape, output_shape):
        """
        Creates a functional model for easier integration with Keras workflows.

        Args:
            input_shape: Shape of the input tensor (batch_size, seq_length, input_dims).
            output_shape: Shape of the output tensor (batch_size, seq_length, output_dims).

        Returns:
            A compiled Keras model.
        """
        inputs = tf.keras.Input(shape=input_shape[1:], name='encoder_input')
        outputs = tf.keras.Input(shape=output_shape[1:], name='decoder_input')

        model = tf.keras.Model(inputs=[inputs, outputs], outputs=self.call(inputs, outputs))
        return model

    # Loss function for CLM and MLM
    def hybrid_loss(self, real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

        # Create mask for padding tokens (assuming 0 is padding token)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=tf.float32)

        mlm_loss = loss_object(real, pred)
        mlm_loss *= mask
        mlm_loss = tf.reduce_sum(mlm_loss) / tf.reduce_sum(mask)

        clm_loss = loss_object(real[:, :-1], pred[:, 1:])
        clm_loss = tf.reduce_mean(clm_loss)

        return 0.3*mlm_loss + 0.7*clm_loss

        #Masked Loss function
        # mask = tf.math.logical_not(tf.math.equal(real, 0))
        # loss = loss_object(real, pred)
        # mask = tf.cast(mask, dtype=loss.dtype)
        # loss *= mask
        # return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # Accuracy with padding mask
    def masked_accuracy(self, real, pred):
        pred = tf.argmax(pred, axis=-1)
        matches = tf.equal(real, tf.cast(pred, tf.int32))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        matches = tf.math.logical_and(matches, mask)
        return tf.reduce_sum(tf.cast(matches, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))

    # Training step
    @tf.function
    def train_step(self, inp, tar, optimizer):
        with tf.GradientTape() as tape:
            predictions = self.call(inp, tar, training=True)
            loss = self.masked_loss(tar[:, 1:], predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

def train(self, train_data, val_data, epochs=50, batch_size=64, d_model=128, warmup_steps=4000, patience=5, save_every=5):
        # Use custom learning rate schedule
        learning_rate = TransformerLRSchedule(d_model=d_model, warmup_steps=warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            total_loss, total_acc = 0, 0
            steps = 0

            # Training loop
            for (inputs, outputs) in train_data.batch(batch_size):
                loss, acc = self.train_step(inputs, outputs, optimizer)
                total_loss += loss
                total_acc += acc
                steps += 1

            train_loss = total_loss / steps
            train_acc = total_acc / steps

            # Validation loop
            val_loss, val_acc = 0, 0
            steps = 0
            for (val_inputs, val_outputs) in val_data.batch(batch_size):
                predictions = self.call(val_inputs, val_outputs[:, :-1])
                val_loss += self.hybrid_loss(val_outputs[:, 1:], predictions)
                val_acc += self.masked_accuracy(val_outputs[:, 1:], predictions)
                steps += 1

            val_loss /= steps
            val_acc /= steps

            print(f"Epoch {epoch+1}/{epochs} — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_weights('best_transformer_model.h5')
                patience_counter = 0
            else:
                patience_counter += 1

            # Save periodic checkpoint every "save_every" epochs
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"transformer_checkpoint_epoch_{epoch+1}.h5"
                self.save_weights(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model weights
        self.load_weights('best_transformer_model.h5')



class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)