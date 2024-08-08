import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=groq_api_key)

# Initialize the Hugging Face model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_huggingface_embedding(text):
    embedding = embedding_model.encode(text)
    print("==== Generating embeddings... ====")
    return embedding

# Example chat completion request using Groq API
def get_groq_response(text):
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content

# Example usage
text = "What is the capital of France?"
embedding = get_huggingface_embedding(text)
print(embedding)
groq_response = get_groq_response(text)
print(groq_response)
