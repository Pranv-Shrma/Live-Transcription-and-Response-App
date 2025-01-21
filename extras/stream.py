# streamlit_app.py
import streamlit as st
import asyncio
import threading
import queue
import websockets
from main_client import receive_messages, AUDIO_SERVER_URL  # Import necessary items

# Queue for messages
message_queue = queue.Queue()

async def streamlit_receive_messages(websocket): # Modified receive_messages for Streamlit
    try:
        while True:
            response1 = await websocket.recv()
            response2 = await websocket.recv()
            print(f"Received transcription from server: {response1}")
            print(f"Received LLM Response from server: {response2}")
            message_queue.put({"role": "user", "content": response1})
            message_queue.put({"role": "assistant", "content": response2})

    except websockets.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error receiving message: {e}")
        message_queue.put({"role": "system", "content": f"Error: {e}"}) # Put error messages in the queue


async def connect_and_receive():
    async with websockets.connect(AUDIO_SERVER_URL) as websocket:
        await streamlit_receive_messages(websocket)


def display_messages(q):
    while True:
        try:
            message = q.get(True, timeout=1)
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        except queue.Empty:
            pass
        except Exception as e:
            st.error(f"Error displaying message: {e}")
            break


# Initialize chat history (if needed - you might remove this if not using persistent history)
if "messages" not in st.session_state:
    st.session_state.message = []


# Start the message display and websocket receiver threads
display_thread = threading.Thread(target=display_messages, args=(message_queue,), daemon=True)
display_thread.start()

asyncio.run(connect_and_receive()) # Start receiving messages

# Rest of your Streamlit app code (if any)
# ...