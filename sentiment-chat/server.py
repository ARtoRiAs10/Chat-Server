# server.py

import asyncio
import json
from datetime import datetime
from transformers import pipeline

# --- Global State ---
clients = {}  # Using a dictionary to store writer and username: {writer: "username"}
ML_MODELS = {} # Lazy-load models into this dictionary

def get_model(model_name: str, task: str, model_id: str):
    """Lazily loads and returns a Hugging Face pipeline."""
    if model_name not in ML_MODELS:
        print(f"Loading model '{model_name}' for task '{task}'... This may take a moment.")
        try:
            ML_MODELS[model_name] = pipeline(task, model=model_id)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    return ML_MODELS.get(model_name)

async def broadcast(message: dict, sender_writer: asyncio.StreamWriter):
    """Encodes a message to JSON and sends it to all connected clients except the sender."""
    # We only broadcast chat messages, not server responses to specific commands
    if message.get("type") != "chat_message":
        return

    # Create a list of writers to avoid issues if the dictionary changes during iteration
    all_writers = list(clients.keys())
    for writer in all_writers:
        if writer != sender_writer:
            try:
                # Add a newline as a delimiter for our JSON messages
                writer.write((json.dumps(message) + '\n').encode())
                await writer.drain()
            except ConnectionError:
                print(f"Client {clients.get(writer, 'Unknown')} disconnected. Removing.")
                del clients[writer]

async def send_direct_message(message: dict, writer: asyncio.StreamWriter):
    """Sends a direct message (like a response or error) to a single client."""
    try:
        writer.write((json.dumps(message) + '\n').encode())
        await writer.drain()
    except ConnectionError:
        print(f"Could not send direct message to {clients.get(writer, 'Unknown')}, they disconnected.")
        if writer in clients:
            del clients[writer]


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Coroutine executed for each client connection."""
    addr = writer.get_extra_info('peername')
    print(f"New connection from {addr}")
    
    # The first message from a client must be a login message
    try:
        login_data = await reader.readline()
        login_info = json.loads(login_data.decode())
        if login_info.get("type") == "login":
            username = login_info.get("username", f"user_{addr[1]}")
            clients[writer] = username
            print(f"{addr} has logged in as {username}")
            
            # Notify the new user they've connected
            await send_direct_message({
                "type": "server_notification",
                "message": f"Welcome, {username}! You are connected to the ML Chat Hub."
            }, writer)
            
            # Notify all other users
            await broadcast({
                "type": "server_notification",
                "message": f"--- {username} has joined the chat ---"
            }, writer) # Use broadcast to send to others
        else:
            raise ValueError("First message was not a valid login.")

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Failed login attempt from {addr}: {e}")
        error_msg = {"type": "error", "message": "Invalid login. Please send JSON with 'type': 'login' and 'username'."}
        await send_direct_message(error_msg, writer)
        writer.close(); await writer.wait_closed(); return
    

    # --- Main message loop ---
    try:
        while True:
            data = await reader.readline()
            if not data:
                break

            try:
                request = json.loads(data.decode())
                message_text = request.get("message", "").strip()
                
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "username": username
                }

                # --- Command Dispatcher ---
                if message_text.startswith("!translate"):
                    parts = message_text.split(" ", 2)
                    if len(parts) < 3:
                        response.update({"type": "error", "message": "Usage: !translate <lang_code> <text>"})
                    else:
                        lang = parts[1]
                        text_to_translate = parts[2]
                        translator = get_model(f"translator_en_{lang}", "translation", f"Helsinki-NLP/opus-mt-en-{lang}")
                        if translator:
                            result = translator(text_to_translate)[0]['translation_text']
                            response.update({"type": "server_response", "message": f"Translation (en->{lang}): {result}"})
                        else:
                            response.update({"type": "error", "message": f"Translation model for '{lang}' not available."})
                    await send_direct_message(response, writer)

                elif message_text.startswith("!generate"):
                    prompt = message_text.replace("!generate", "", 1).strip()
                    generator = get_model("generator", "text-generation", "distilgpt2")
                    if generator and prompt:
                        result = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
                        response.update({"type": "server_response", "message": f"Generated Text: {result}"})
                    else:
                        response.update({"type": "error", "message": "Usage: !generate <your prompt>"})
                    await send_direct_message(response, writer)
                
                elif message_text.startswith("!ner"):
                    text = message_text.replace("!ner", "", 1).strip()
                    ner_model = get_model("ner", "ner", "dbmdz/bert-large-cased-finetuned-conll03-english")
                    if ner_model and text:
                        entities = ner_model(text)
                        # Filter out entities with low scores and format the result
                        filtered_entities = [f"{e['word']} ({e['entity_group']})" for e in entities if e['score'] > 0.85]
                        result_text = ", ".join(filtered_entities) if filtered_entities else "No entities found."
                        response.update({"type": "server_response", "message": f"Entities Found: {result_text}"})
                    else:
                        response.update({"type": "error", "message": "Usage: !ner <your text>"})
                    await send_direct_message(response, writer)

                else: # Default behavior: Sentiment Analysis Chat
                    sentiment_analyzer = get_model("sentiment", "sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
                    sentiment = sentiment_analyzer(message_text)[0]
                    response.update({
                        "type": "chat_message",
                        "message": message_text,
                        "sentiment": {"label": sentiment['label'], "score": round(sentiment['score'], 2)}
                    })
                    # Broadcast regular chat messages to everyone
                    # Send it to the original sender too so they have a record
                    await send_direct_message(response, writer) 
                    await broadcast(response, writer)


            except json.JSONDecodeError:
                await send_direct_message({"type": "error", "message": "Received malformed JSON."}, writer)
            except Exception as e:
                print(f"An error occurred with client {username}: {e}")
                await send_direct_message({"type": "error", "message": f"An internal server error occurred: {e}"}, writer)


    except asyncio.CancelledError:
        pass # Task was cancelled, normal shutdown
    finally:
        print(f"Connection from {username} ({addr}) closed.")
        del clients[writer]
        writer.close()
        await writer.wait_closed()
        await broadcast({
            "type": "server_notification",
            "message": f"--- {clients.get(writer, 'A user')} has left the chat ---"
        }, writer)


async def main():
    host = '127.0.0.1'
    port = 8888
    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    print("ML Chat Hub is running. Waiting for connections...")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        # Pre-load the most common model to make the first chat message faster
        get_model("sentiment", "sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")