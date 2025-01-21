import asyncio
import websockets
import logging
from v1_audio_recording import record_audio  # Updated async generator function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUDIO_SERVER_URL = 'ws://localhost:8080/ws/audio'

async def main():
    async with websockets.connect(AUDIO_SERVER_URL) as websocket:
        audio_file = record_audio()
        await websocket.send(audio_file)
        print(type(audio_file))
        print("Audio chunk sent to the server")
        response1 = await websocket.recv()
        response2 = await websocket.recv()
        print(f"Received transcription from server: {response1}")
        print(f"Received LLM Response from server: {response2}")

if __name__ == "__main__":
    asyncio.run(main())
