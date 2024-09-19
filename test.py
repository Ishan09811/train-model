import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="/home/runner/work/train-model/train-model/chat_model_large.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tokenizer settings (must be the same as used in training)
vocab_size = 10000
max_len_input = 50  # The max length used during training

# Example of a trained tokenizer (this should match your training tokenizer)
tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False, split=' ')
# You need to ensure that the tokenizer is loaded with the same vocabulary
# tokenizer.fit_on_texts(input_texts + target_texts)  # Done during training

# Input "hi" to be tokenized and padded
input_text = "hi"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_padded = pad_sequences(input_sequence, maxlen=max_len_input, padding='post')

# Convert the input to the correct format
input_data = np.array(input_padded, dtype=np.float32)

# Assuming input_data is a sequence with shape (50,)
input_data = np.expand_dims(input_data, axis=0)  # Reshape to (1, 50)

# Set the input tensor for the model
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor (model's response in token form)
output_data = interpreter.get_tensor(output_details[0]['index'])

# The model's output will be token indices, let's print them
print("Chatbot response (token indices):", np.argmax(output_data, axis=-1))

# Reverse mapping from token indices to words
reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}

# Convert the output tokens back to words
output_tokens = np.argmax(output_data, axis=-1)[0]
response = ' '.join([reverse_word_map.get(token, '') for token in output_tokens if token != 0])
print("Chatbot response (text):", response)
