import threading
import time
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server, A2AClient

# ------------------ SERVER ------------------
class EchoAgent(A2AServer):
    def handle_message(self, message):
        print(f"[Server] Received: {message.content.text}")
        return Message(
            content=TextContent(text=f"Echo: {message.content.text}"),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

def start_server():
    agent = EchoAgent()
    run_server(agent, host="127.0.0.1", port=5055)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# ------------------ CLIENT ------------------
time.sleep(2)  # Allow server to start

try:
    client = A2AClient("http://localhost:5055/a2a")
    message = Message(
        content=TextContent(text="Hello A2A from same file!"),
        role=MessageRole.USER
    )
    response = client.send_message(message)
    print(f"[Client] Received: {response.content.text}")
except Exception as e:
    print(f"[Client] Error: {e}")
