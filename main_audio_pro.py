import asyncio
import numpy as np
import tensorflow as tf
from typing import AsyncGenerator
import zipfile


# Audio Configuration 
CHANNELS = 1
RATE = 16000
CHUNK = 1024
TARGET_LENGTH = 15600

# Loading TFLite model
model_path = '1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
waveform_input_index = input_details[0]['index']
scores_output_index = output_details[0]['index']
interpreter.resize_tensor_input(waveform_input_index, [TARGET_LENGTH], strict=False)
interpreter.allocate_tensors()

# labels of model output 
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
    recording = False
    silence_counter = 0
    silence_threshold = 30  # A silence threshold for stopping further audio processing until speech is detected again
    speech_confidence_threshold = 0.7 # A confidence threshold for speech confirmation

    async for audio_chunk in audio_chunk_generator:
        try:
            audio_data_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0
            buffered_samples = np.append(buffered_samples, audio_data_float32)
            
            if len(buffered_samples) >= TARGET_LENGTH:
                interpreter.set_tensor(waveform_input_index, buffered_samples[:TARGET_LENGTH])
                interpreter.invoke()
                scores = interpreter.get_tensor(scores_output_index)
                top_class_index = scores.argmax()
                prediction = labels[top_class_index]
                speech_score = scores[0][top_class_index] # Get the confidence score


                if prediction == "Speech" and speech_score >= speech_confidence_threshold:
                    if not recording:  # Start recording only if not already recording
                        recording = True
                    silence_counter = 0
                    buffered_audio.extend(audio_chunk)  # Append only if speech is detected with confidence

                elif recording:
                    silence_counter += 1
                    if silence_counter >= silence_threshold:
                        if buffered_audio: # Yielding data only if buffered audio is not empty
                            yield bytes(buffered_audio)
                        buffered_audio.clear()
                        recording = False
                        silence_counter = 0
                buffered_samples = buffered_samples[CHUNK:]  # Removing already processed samples


        except Exception as e:
            print(f"Error during audio processing: {e}")

    # Yield remaining audio only if recording was active and buffer is not empty
    if recording and buffered_audio: 
        yield bytes(buffered_audio)