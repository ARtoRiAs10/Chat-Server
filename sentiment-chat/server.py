# server.py

import asyncio
from transformers import pipeline

# --- Global State ---
# A set to store all connected client writer objects
clients = set()
# Load the sentiment analysis model once when the server starts.
# The first time this runs, it will download the model (a few hundred MB).
print("Loading sentiment analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded.")

# --- Server Logic ---

async def broadcast(message: str, sender_writer: asyncio.StreamWriter):
    """
    Sends a message to all connected clients except the sender.
    """
    # Create a list of writers to send messages to
    # We iterate over a copy of the set in case it's modified during iteration
    for client_writer in list(clients):
        if client_writer != sender_writer:
            try:
                client_writer.write(message.encode())
                await client_writer.drain()
            except ConnectionError:
                # If the client is disconnected, remove them
                print("A client disconnected, removing from list.")
                clients.remove(client_writer)

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """
    This coroutine is executed for each new client connection.
    """
    # Add the new client's writer to our global set
    clients.add(writer)
    addr = writer.get_extra_info('peername')
    print(f"New connection from {addr}")

    try:
        while True:
            # Wait for a message from the client
            # The '1024' is the buffer size in bytes
            data = await reader.read(1024)
            if not data:
                # An empty byte string means the client disconnected
                break

            message = data.decode().strip()
            print(f"Received from {addr}: {message}")

            # 1. Perform Sentiment Analysis
            try:
                result = sentiment_analyzer(message)[0]
                sentiment = result['label']
                score = result['score']
                formatted_message = f"[{addr[0]}:{addr[1]}]: {message} (SENTIMENT: {sentiment}, Score: {score:.2f})\n"
            except Exception as e:
                print(f"Error during sentiment analysis: {e}")
                formatted_message = f"[{addr[0]}:{addr[1]}]: {message} (Analysis failed)\n"


            # 2. Broadcast the message with sentiment to other clients
            print(f"Broadcasting: {formatted_message.strip()}")
            await broadcast(formatted_message, writer)

    except asyncio.CancelledError:
        # This happens when the server is shutting down
        pass
    except ConnectionResetError:
        print(f"Connection reset by {addr}")
    finally:
        # When the loop is broken, the client has disconnected.
        print(f"Connection from {addr} closed.")
        clients.remove(writer)
        writer.close()
        await writer.wait_closed()
        await broadcast(f"--- Client {addr} has left the chat ---\n", writer)


async def main():
    """
    The main function to start the server.
    """
    host = '127.0.0.1'  # localhost
    port = 8888

    # Start the server
    server = await asyncio.start_server(handle_client, host, port)

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    # Keep the server running
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")