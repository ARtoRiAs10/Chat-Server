# client.py (Upgraded for JSON and Usernames)

import asyncio
import json
import aioconsole  # A library for async input, install with: pip install aioconsole

async def receive_messages(reader: asyncio.StreamReader):
    """Listens for JSON messages from the server and prints them."""
    while True:
        try:
            data = await reader.readline()
            if not data:
                print("\n[Connection to server lost.]")
                break
            
            try:
                msg_data = json.loads(data.decode())
                msg_type = msg_data.get("type")
                message = msg_data.get("message", "")
                username = msg_data.get("username", "Server")

                if msg_type == "chat_message":
                    sentiment = msg_data.get("sentiment", {})
                    print(f"\n<{username}> {message}  ({sentiment.get('label')})")
                elif msg_type in ["server_notification", "server_response", "error"]:
                    print(f"\n[{username.upper()}] {message}")
                else:
                    print(f"\n{str(msg_data)}")

            except json.JSONDecodeError:
                print(f"\n[Received non-JSON data: {data.decode().strip()}]")

        except (ConnectionResetError, asyncio.CancelledError):
            print("\n[Disconnected from the server.]")
            break

async def send_messages(writer: asyncio.StreamWriter, username: str):
    """Gets user input asynchronously and sends it to the server as JSON."""
    # First, send the login message
    login_msg = json.dumps({"type": "login", "username": username}) + '\n'
    writer.write(login_msg.encode())
    await writer.drain()
    
    while True:
        try:
            # aioconsole.ainput allows for non-blocking input
            message_text = await aioconsole.ainput(f'You ({username}): ')
            if message_text.lower() == 'exit':
                break
            
            payload = json.dumps({"message": message_text}) + '\n'
            writer.write(payload.encode())
            await writer.drain()
        except (EOFError, ConnectionResetError, asyncio.CancelledError):
            break
    
    writer.close()
    await writer.wait_closed()

async def main():
    """Main function to connect to the server and manage tasks."""
    host = '127.0.0.1'
    port = 8888

    try:
        reader, writer = await asyncio.open_connection(host, port)
    except ConnectionRefusedError:
        print("Connection failed. Is the server running?")
        return
        
    username = await aioconsole.ainput("Enter your username: ")

    receive_task = asyncio.create_task(receive_messages(reader))
    send_task = asyncio.create_task(send_messages(writer, username))

    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

if __name__ == "__main__":
    try:
        # You may need to install aioconsole first: pip install aioconsole
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutting down.")