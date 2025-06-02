import threading
import time
from python_a2a.mcp import MCPEnabledAgent
from python_a2a import (
    run_server,
    Message,
    TextContent,
    MessageRole,
    A2AClient
)

# ------------------ MCP-Enabled Agent ------------------
class SummarizerMCPAgent(MCPEnabledAgent):
    def __init__(self):
        super().__init__()
        self.name = "summarizer-mcp-agent"
        self.description = "An MCP-compliant agent that summarizes text."

    def handle_message(self, message: Message):
        print(f"[Agent] Received: {message.content.text}")
        if "summarize" in message.content.text.lower():
            return Message(
                content=TextContent(text="📝 Summary: This is a short version of your text. as an example"),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )
        return Message(
            content=TextContent(text="Please include 'summarize' to trigger the summarizer."),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id
        )

# ------------------ Server Thread ------------------
def start_server():
    agent = SummarizerMCPAgent()
    run_server(agent, host="127.0.0.1", port=6061)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(2)  # Give the server time to start

# ------------------ Client Request ------------------
client = A2AClient("http://localhost:6061/a2a")
message = Message(
    content=TextContent(text="Can you summarize this for me?"),
    role=MessageRole.USER
)

try:
    response = client.send_message(message)
    print(f"[Client] Received: {response.content.text}")
except Exception as e:
    print(f"[Client] Error: {e}")
