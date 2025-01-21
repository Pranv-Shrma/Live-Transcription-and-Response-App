import pyaudio
import numpy as np
import tensorflow as tf
import zipfile
import asyncio

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
TARGET_LENGTH = 15600

# Load the TFLite model for speech detection
model_path = '1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
waveform_input_index = input_details[0]['index']
scores_output_index = output_details[0]['index']
interpreter.resize_tensor_input(waveform_input_index, [TARGET_LENGTH], strict=False)
interpreter.allocate_tensors()

async def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    audio_buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)
    recording_duration = 5  # Set minimum recording duration (in seconds) for testing

    print("Listening... Starting async record_audio function")

    try:
        with zipfile.ZipFile(model_path) as z:
            with z.open('yamnet_label_list.txt') as f:
                labels = [line.decode('utf-8').strip() for line in f]

        start_time = asyncio.get_event_loop().time()

        while True:
            # Enforce a minimum recording duration
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > recording_duration:
                print("Minimum recording duration reached.")
                break

            # Read audio data and add it to the buffer
            audio_data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(audio_chunk))
            audio_buffer[-len(audio_chunk):] = audio_chunk

            # Run model and get prediction (but ignore it temporarily)
            interpreter.set_tensor(input_details[0]['index'], audio_buffer)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])
            prediction = labels[scores.argmax()]
            print(f"Prediction: {prediction}")

            # Yield audio data regardless of prediction for testing purposes
            yield audio_data

            # Small delay to keep the loop non-blocking
            await asyncio.sleep(0.01)

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
