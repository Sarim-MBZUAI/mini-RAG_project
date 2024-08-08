import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

sentence_transformer_ef = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

def get_huggingface_embedding(text):
    return sentence_transformer_ef([text])[0]

for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_huggingface_embedding(doc["text"])

for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

def query_documents(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = results["documents"][0]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer

question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)