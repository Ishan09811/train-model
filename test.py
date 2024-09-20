import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="/home/runner/work/train-model/train-model/chat_model_large.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input "hi" to be tokenized and padded
input_text = "hi"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_padded = pad_sequences(input_sequence, maxlen=1, padding='post')  # Change maxlen to 1

# Convert the input to the correct format
input_data = np.array(input_padded, dtype=np.float32)  # Shape will be (1, 1)

# Set the input tensor for the model
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor (model's response in token form)
output_data = interpreter.get_tensor(output_details[0]['index'])

# Convert output tokens back to text
output_tokens = np.argmax(output_data, axis=-1)[0]
reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}
response = ' '.join([reverse_word_map.get(token, '') for token in output_tokens if token != 0])

print("Chatbot response (text):", response)
