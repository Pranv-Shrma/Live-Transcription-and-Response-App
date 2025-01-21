# audio_processor.py
import asyncio
import numpy as np
import tensorflow as tf
from typing import AsyncGenerator
import zipfile

# Audio Configuration (must match client and server)
CHANNELS = 1
RATE = 16000
CHUNK = 1024
TARGET_LENGTH = 15600

# Load your TFLite model
model_path = '1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
waveform_input_index = input_details[0]['index']
scores_output_index = output_details[0]['index']
interpreter.resize_tensor_input(waveform_input_index, [TARGET_LENGTH], strict=False)
interpreter.allocate_tensors()


try:
    with zipfile.ZipFile(model_path) as z:
        with z.open('yamnet_label_list.txt') as f:
            labels = [line.decode('utf-8').strip() for line in f]
except zipfile.BadZipFile: # if not a zipfile, try directly opening
    with open('yamnet_label_list.txt', 'r') as f:
        labels = [line.strip() for line in f]


async def process_audio(audio_chunk_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    buffered_audio = bytearray()
    buffered_samples = np.array([], dtype=np.float32)
    recording = False  # Flag to indicate if recording
    silence_counter = 0
    silence_threshold = 15  # Number of silent chunks before stopping recording

    async for audio_chunk in audio_chunk_generator:
        try:
            # ... (audio processing and speech detection as before)
            audio_data_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0
            buffered_samples = np.append(buffered_samples, audio_data_float32)
            buffered_audio.extend(audio_chunk)

            if len(buffered_samples) >= TARGET_LENGTH:
                interpreter.set_tensor(waveform_input_index, buffered_samples[:TARGET_LENGTH])
                interpreter.invoke()
                scores = interpreter.get_tensor(scores_output_index)
                top_class_index = scores.argmax()
                prediction = labels[top_class_index]  # Assuming 'labels' is defined

                if prediction == "Speech":  # Or your speech label
                    recording = True
                    silence_counter = 0
                    buffered_audio.extend(audio_chunk)
                elif recording:  # Only count silence if already recording
                    silence_counter += 1
                    if silence_counter >= silence_threshold:
                        yield bytes(buffered_audio)
                        buffered_audio.clear()
                        buffered_samples = buffered_samples[TARGET_LENGTH:]
                        recording = False  # Stop recording
                        silence_counter = 0 # reset silence counter
                buffered_samples = buffered_samples[CHUNK:] # Remove processed samples


        except Exception as e:
            print(f"Error during audio processing: {e}")

    # Yield any remaining audio after the loop finishes
    if buffered_audio:
        yield bytes(buffered_audio)