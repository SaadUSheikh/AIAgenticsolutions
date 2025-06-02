#AI10

import uuid
import threading
import time
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server, A2AClient

# ------------------ ACP Agent ------------------
class ACPAgent(A2AServer):
    def handle_message(self, message: Message) -> Message:
        print(f"[Server] Received from {message.role}: {message.content.text}")
        return Message(
            content=TextContent(text="🤖 Hello, I received your message."),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

# ------------------ Start Server Thread ------------------
def start_server():
    agent = ACPAgent()
    run_server(agent, host="127.0.0.1", port=7070)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(2)  # Let server start

# ------------------ ACP Client ------------------
client = A2AClient("http://localhost:7070/a2a")

message = Message(
    content=TextContent(text="Hi agent, can you hear me?"),
    role=MessageRole.USER,
    message_id=str(uuid.uuid4()),
    conversation_id=str(uuid.uuid4())
)

response = client.send_message(message)
print(f"[Client] Got response from {response.role}: {response.content.text}")
