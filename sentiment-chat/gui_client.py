# gui_client.py

import tkinter as tk
from tkinter import scrolledtext, simpledialog
import asyncio
import threading
import json
import queue

class ChatGUI:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        
        # Queues for thread-safe communication between Tkinter and asyncio
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = asyncio.Queue()

        # --- Tkinter UI Setup ---
        self.root = tk.Tk()
        self.root.title("ML-Powered Chat")
        self.root.geometry("600x450")

        self.username = simpledialog.askstring("Username", "Please enter your username:", parent=self.root)
        if not self.username:
            self.root.destroy()
            return
        
        self.root.title(f"ML-Powered Chat - {self.username}")

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.pack(padx=10, pady=5, fill=tk.X)

        self.msg_entry = tk.Entry(self.entry_frame, font=("Helvetica", 12))
        self.msg_entry.bind("<Return>", self.send_message)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the networking thread
        self.thread = threading.Thread(target=self.start_asyncio_loop, daemon=True)
        self.thread.start()

        # Start the periodic check for new messages
        self.process_incoming()
        
        self.root.mainloop()

    def start_asyncio_loop(self):
        """Runs the asyncio event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.network_client())
        self.loop.close()

    async def network_client(self):
        """The main asyncio client logic."""
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            
            # Send login message
            login_msg = json.dumps({"type": "login", "username": self.username}) + '\n'
            writer.write(login_msg.encode())
            await writer.drain()

            # Create two concurrent tasks for reading and writing
            reader_task = asyncio.create_task(self.reader_coro(reader))
            writer_task = asyncio.create_task(self.writer_coro(writer))
            
            await asyncio.gather(reader_task, writer_task)

        except ConnectionRefusedError:
            self.display_message("--- Connection failed. Is the server running? ---")
        except Exception as e:
            self.display_message(f"--- An error occurred: {e} ---")
        finally:
            print("Network client loop finished.")

    async def reader_coro(self, reader):
        """Reads messages from the server and puts them in the queue."""
        while True:
            try:
                data = await reader.readline()
                if not data:
                    break
                self.incoming_queue.put(data.decode())
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Reader error: {e}")
                break
        self.incoming_queue.put('{"type": "server_notification", "message": "--- Disconnected from server ---"}')

    async def writer_coro(self, writer):
        """Takes messages from the outgoing queue and sends them to the server."""
        while True:
            try:
                message = await self.outgoing_queue.get()
                writer.write((message + '\n').encode())
                await writer.drain()
                self.outgoing_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Writer error: {e}")
                break
        writer.close()
        await writer.wait_closed()


    def process_incoming(self):
        """Processes messages from the incoming_queue to display on the GUI."""
        while not self.incoming_queue.empty():
            try:
                msg_json_str = self.incoming_queue.get_nowait()
                msg_data = json.loads(msg_json_str)
                self.format_and_display(msg_data)
            except queue.Empty:
                pass
            except json.JSONDecodeError:
                self.display_message(f"--- Received malformed data: {msg_json_str[:100]} ---")
        
        # Schedule the next check
        self.root.after(100, self.process_incoming)

    def format_and_display(self, msg_data):
        """Formats the JSON message for pretty printing in the text area."""
        msg_type = msg_data.get("type")
        message = msg_data.get("message", "")
        username = msg_data.get("username", "Server")
        
        formatted_message = ""
        if msg_type == "chat_message":
            sentiment = msg_data.get("sentiment", {})
            label = sentiment.get('label', 'UNKNOWN')
            score = sentiment.get('score', 0)
            formatted_message = f"<{username}> {message}  ({label}: {score})"
        elif msg_type == "server_notification":
            formatted_message = f"--- {message} ---"
        elif msg_type == "server_response":
            formatted_message = f"[BOT RESPONSE] {message}"
        elif msg_type == "error":
            formatted_message = f"[ERROR] {message}"
        else:
            formatted_message = str(msg_data)
        
        self.display_message(formatted_message)

    def display_message(self, message):
        """Inserts a message into the text area."""
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, message + '\n')
        self.text_area.config(state='disabled')
        self.text_area.yview(tk.END) # Auto-scroll

    def send_message(self, event=None):
        """Sends the message from the entry box."""
        message_text = self.msg_entry.get()
        if message_text:
            payload = json.dumps({"message": message_text})
            self.outgoing_queue.put_nowait(payload)
            self.msg_entry.delete(0, tk.END)

    def on_closing(self):
        """Handles the window close event to shut down gracefully."""
        print("Closing application...")
        if hasattr(self, 'loop') and self.loop.is_running():
            # This is a bit forceful but necessary for cleanup in this threaded model
            for task in asyncio.all_tasks(loop=self.loop):
                task.cancel()
        self.root.destroy()

if __name__ == "__main__":
    ChatGUI()