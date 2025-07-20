from llama_index.core import VectorStoreIndex, Document

# Load content from a local file
with open("saad.txt", "r") as f:
    text = f.read()

# Wrap it in a Document object
doc = Document(text=text)

# Build an in-memory index
index = VectorStoreIndex.from_documents([doc])

# Create a query engine
query_engine = index.as_query_engine()

# Ask a question
response = query_engine.query("Who is Saad? What is his theory?")
print(response)
