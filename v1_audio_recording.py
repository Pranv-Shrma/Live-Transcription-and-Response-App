import pyaudio
import numpy as np
import wave
import tensorflow as tf
import zipfile

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
TARGET_LENGTH = 15600  # Length of buffer to match model input

# Ensure the input tensor is correctly sized
model_path = '1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
waveform_input_index = input_details[0]['index']
scores_output_index = output_details[0]['index']
interpreter.resize_tensor_input(waveform_input_index, [TARGET_LENGTH], strict=False)
interpreter.allocate_tensors()

# Speech Detection Threshold
SILENCE_THRESHOLD = 30  # Number of consecutive "Not Speech" predictions to stop

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    is_recording = False
    audio_frames = []
    audio_buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)  # initializing audio_buffer
    recording_finished = False

    print("Listening... Starting Record Audio Function")

    try:
        with zipfile.ZipFile(model_path) as z:
            with z.open('yamnet_label_list.txt') as f:
                labels = [line.decode('utf-8').strip() for line in f]
        silence_count = 0  # Counter for consecutive "Not Speech" predictions

        while True and not recording_finished:  # Loop until recording is finished
            # Read audio data and convert
            audio_data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(audio_chunk))
            audio_buffer[-len(audio_chunk):] = audio_chunk

            # Run speech detection model
            interpreter.set_tensor(input_details[0]['index'], audio_buffer)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])
            prediction = labels[scores.argmax()]

            if prediction == "Speech" and not is_recording:
                print("Speech detected. Recording...")
                is_recording = True
                silence_count = 0  # Reset silence counter

            if is_recording:
                audio_frames.append(audio_data)
        
            if prediction != "Speech" and is_recording:
                silence_count += 1
                if silence_count >= SILENCE_THRESHOLD:
                    print("Speech ended. Saving audio...")
                    is_recording = False
                # silence_count = 0  # Reset silence counter
                    recording_finished = True  # Set flag to indicate recording completion

                # Save recorded audio to file
                    recorded_audio_path = "recorded_audio.wav" # Specify the desired file path
                    wf = wave.open(recorded_audio_path, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_frames))
                    wf.close()
                    audio_frames = []
                    
                    # return audio_buffer.astype(np.int16).tobytes()
                    
                    with wave.open(recorded_audio_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        return frames
            

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

record_audio()