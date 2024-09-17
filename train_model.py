import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define file path
file_path = 'dialogs.txt'

# Load dataset
df = pd.read_csv(file_path, delimiter='\t', header=None, names=['input_text', 'target_text'])

# Preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove leading and trailing spaces
    return text

df['input_text'] = df['input_text'].apply(clean_text)
df['target_text'] = df['target_text'].apply(clean_text)
df['target_text'] = '\t' + df['target_text'] + '\n'

# Extract lists of texts
input_texts = df['input_text'].tolist()
target_texts = df['target_text'].tolist()

# Define parameters
vocab_size = 10000
max_len_input = 50
max_len_target = 50

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False, split=' ')
tokenizer.fit_on_texts(input_texts + target_texts)

# Convert text sequences to integer sequences
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Padding sequences
encoder_input_data = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# Prepare target data for training (one-hot encoding)
decoder_target_data = np.zeros((len(target_sequences), max_len_target, vocab_size), dtype='float32')

for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq):
        if word_id > 0:
            decoder_target_data[i, t, word_id] = 1.0  # One-hot encode

print("Data preparation completed:")
print(f"Encoder input data shape: {encoder_input_data.shape}")
print(f"Decoder input data shape: {decoder_input_data.shape}")
print(f"Decoder target data shape: {decoder_target_data.shape}")

# Define model
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Build and compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Train the model
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=128,
    epochs=10
)

# Save the model
model.save('chat_model_large.h5')

# Convert to TensorFlow Lite format (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('chat_model_large.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model training and saving completed.")
