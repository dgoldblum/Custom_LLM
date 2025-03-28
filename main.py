import tensorflow as tf
import numpy as np
import os
import transformer_arch as arch
from transformer_arch import Tokenizer
from transformer import Transformer
import dask.dataframe as dd
from datasets import load_dataset



print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))
if not tf.config.experimental.list_physical_devices('GPU'):
    raise RuntimeError("No GPU detected! Ensure that ROCm and the correct drivers are installed.")

# Set mixed precision for better performance
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Training Parameters
INPUT_DIMS = 128
OUTPUT_DIMS = 128
N_ENC = 6
N_DEC = 6
D_MODEL = 128
N_HEADS = 8
FF_DIM = 2048
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 5e-4

#f = dd.read_parquet("hf://datasets/Locutusque/UltraTextbooks/data/train-*.parquet")


dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_texts = dataset["train"]["text"]
val_texts = dataset["validation"]["text"]

checkpoint_dir = './checkpoints'
#dataset_path = './data/'

tokenizer = Tokenizer(max_length=128)

def prepare_data(texts):
    inputs, outputs = [], []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue  # Skip too short
        inputs.append(token_ids[:-1])   # e.g., [CLS] A cat sat
        outputs.append(token_ids[1:])   # e.g.,     A cat sat [EOS]
    return tf.data.Dataset.from_tensor_slices((inputs, outputs))

train_dataset = prepare_data(train_texts).shuffle(1000).batch(64)
val_dataset = prepare_data(val_texts).batch(64)

transformer = Transformer(
    input_dims=INPUT_DIMS,
    output_dims=OUTPUT_DIMS,
    n_enc=N_ENC,
    n_dec=N_DEC,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    ff_dim=FF_DIM
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

loss_fn = transformer.hybrid_loss
metric = transformer.masked_accuracy

model = transformer.create_model(
    input_shape=train_dataset[0].shape, output_shape=train_dataset[1].shape
)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "transformer_epoch_{epoch:02d}.h5"),
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

# Training Loop
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_callback]
)

model.save(os.path.join(checkpoint_dir, "final_model.h5"))
