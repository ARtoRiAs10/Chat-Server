# client.py

import asyncio

async def receive_messages(reader: asyncio.StreamReader):
    """
    Listens for messages from the server and prints them.
    """
    while True:
        try:
            data = await reader.read(1024)
            if not data:
                print("\nConnection to server lost.")
                break
            message = data.decode()
            # We use print with end='' and flush to avoid conflicts with input()
            print(f"\r{message.strip()}\nYou: ", end='', flush=True)
        except (ConnectionResetError, asyncio.CancelledError):
            print("\nDisconnected from the server.")
            break

async def send_messages(writer: asyncio.StreamWriter):
    """
    Gets user input from the command line and sends it to the server.
    """
    # Use asyncio's to_thread to run the blocking input() in a separate thread
    loop = asyncio.get_running_loop()
    while True:
        try:
            message = await loop.run_in_executor(None, input, "You: ")
            if message.lower() == 'exit':
                break
            writer.write(message.encode())
            await writer.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            break
    
    writer.close()
    await writer.wait_closed()


async def main():
    """
    Main function to connect to the server and manage tasks.
    """
    host = '127.0.0.1'
    port = 8888

    try:
        reader, writer = await asyncio.open_connection(host, port)
    except ConnectionRefusedError:
        print("Connection failed. Is the server running?")
        return

    # Create two tasks that run concurrently
    receive_task = asyncio.create_task(receive_messages(reader))
    send_task = asyncio.create_task(send_messages(writer))

    # Wait for either task to complete (which happens on disconnection or 'exit')
    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the other task to ensure a clean shutdown
    for task in pending:
        task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutting down.")