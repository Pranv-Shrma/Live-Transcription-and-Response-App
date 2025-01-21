from openai import OpenAI
import websockets
from fastapi import FastAPI, WebSocket
import numpy as np
import logging
import wave
from pathlib import Path
from dotenv import load_dotenv
import os



# from audio_recording import record_audio

# Initialize the FastAPI app
load_dotenv()
app = FastAPI()
api_key = os.getenv('API_KEY')

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key = api_key
)

def bytes_to_wav(audio_bytes, file_path):
    sample_rate=16000 
    channels=1 
    sample_width=2
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)  


async def transcribe_audio(audio_data):
    """Send audio to Whisper API for transcription."""
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_data
    )
    return response.text

async def get_llm_response(transcription):
    """Send transcription to LLM (ChatGPT) for a response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message.content


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            audio = await websocket.receive_bytes()
            bytes_to_wav(audio,'server_records/recording.wav')
            audio_file=Path("server_records/recording.wav")
            
            
            logging.info("Transcribing audio...")
            transcription = await transcribe_audio(audio_file)
            await websocket.send_text(transcription)

            logging.info(f"Transcription result: {transcription}")
            
            # Send transcription to LLM for a response
            logging.info("Generating LLM response...")
            llm_response = await get_llm_response(transcription)

            # Send the LLM response back to the client
            await websocket.send_text(llm_response)
            logging.info(f"LLM Response: {llm_response}")


    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")


    await websocket.close()

