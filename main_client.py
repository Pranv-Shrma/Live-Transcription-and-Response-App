import asyncio
import websockets
import pyaudio
import threading
import logging
import json
import time
import struct


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
AUDIO_SERVER_URL = 'ws://localhost:8000/ws/audio' # Websocket URL

# send audio to the server 
async def audio_sender(queue, websocket):
    while True:
        audio_data = await queue.get()
        await websocket.send(audio_data)

# audio in queue
async def put_to_queue(queue, data):
    await queue.put(data)

# client side recording function
def record_audio_to_queue(queue, loop):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    try:
        while True:
            data = stream.read(CHUNK)
            asyncio.run_coroutine_threadsafe(put_to_queue(queue, data), loop)
    except Exception as e:
        print(f"Error recording audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


# client receiving response from server
async def receive_messages(websocket):
    try:
        while True:
            response1 = await websocket.recv()
            response2 = await websocket.recv()
            print(f"Received transcription from server: {response1}")
            print(f"Received LLM Response from server: {response2}")

    except websockets.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error receiving message: {e}")



async def main():
    async with websockets.connect(AUDIO_SERVER_URL) as websocket:
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        audio_thread = threading.Thread(target=record_audio_to_queue, args=(queue, loop))
        audio_thread.start()
            
        try:
            await asyncio.gather(
                audio_sender(queue, websocket),
                receive_messages(websocket),
            )
        except Exception as e:
            logging.error(f"Error in main: {e}")
        finally:
            audio_thread.join()

if __name__ == "__main__":
    asyncio.run(main())