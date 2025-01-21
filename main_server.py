# server.py
import asyncio
import websockets
from fastapi import FastAPI, WebSocket
import io
import wave
from openai import OpenAI
import numpy as np
from typing import AsyncGenerator
from dotenv import load_dotenv
import json
import os

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
api_key = os.getenv('API_KEY')
# Now import TensorFlow
import tensorflow as tf


# from main_audio_processor import process_audio  
from main_audio_pro import process_audio # My audio processing file

app = FastAPI()
client = OpenAI(api_key=api_key)

# Audio Configuration 
CHANNELS = 1
RATE = 16000

async def transcribe_audio(audio_data):
    response = client.audio.transcriptions.create(
        model="whisper-1", file=audio_data, language="en"
    )
    return response.text

# async def get_llm_response(transcription):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",  # Or your preferred model
#         messages=[{"role": "user", "content": transcription}],
#     )
#     return response.choices[0].message.content
async def get_llm_response(transcription):
    response = client.chat.completions.create(
        model="gpt-4-0613",  # Use a model with tool-calling capabilities
        messages=[{"role": "user", "content": transcription}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generates an image based on a text prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The text prompt to generate the image from."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ],
        tool_choice="auto"  # Let the model decide whether to use a tool
    )

    message = response.choices[0].message
    if message.tool_calls:
        # The model decided to use a tool (generate_image)
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "generate_image":
            image_prompt = json.loads(tool_call.function.arguments).get("prompt")
            print(f"Generating image for prompt: {image_prompt}")
            image_response = client.images.generate(
                prompt=image_prompt,
                n=1,
                size="256x256"  # Adjust size as needed
            )
            image_url = image_response.data[0].url
            final_response = f"Generated image: {image_url}" # Return the image URL
        else:
            final_response = "Unknown tool call." # Handle unexpected tool calls
    else:
        # The model provided a textual response directly
        final_response = message.content

    return final_response

# Main server code
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    async def audio_chunk_generator():
        while True:
            try:
                audio_chunk = await websocket.receive_bytes()
                yield audio_chunk
            except websockets.ConnectionClosedOK:
                break
            except Exception as e:
                print(f"Error receiving audio: {e}")
                break

    try:
        async for audio_data in process_audio(audio_chunk_generator()):
            if audio_data:
                with io.BytesIO() as wav_buffer:
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(RATE)
                        wf.writeframes(audio_data)
                    wav_buffer.seek(0)
                    wav_file = io.BytesIO(wav_buffer.read())
                    wav_file.name = "audio.wav"

                transcription = await transcribe_audio(wav_file)
                print(f"Transcription: {transcription}")
                await websocket.send_text(transcription)

                llm_response = await get_llm_response(transcription)
                print(f"LLM Response: {llm_response}")
                await websocket.send_text(llm_response)



    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()