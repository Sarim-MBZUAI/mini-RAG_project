# RAG (Retrieval-Augmented Generation) Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage](#usage)
7. [How It Works](#how-it-works)
8. [Customization](#customization)
9. [Advanced Usage](#advanced-usage)
10. [Performance Considerations](#performance-considerations)
11. [Contributing](#contributing)
12. [License](#license)


![RAG Architecture](https://github.com/Sarim-MBZUAI/mini-RAG_project/blob/main/asset/full_naive_rag.png)

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) system. RAG is a hybrid approach that combines the strengths of retrieval-based and generation-based models in natural language processing. It enhances the capabilities of large language models by providing them with relevant information retrieved from a knowledge base, allowing for more accurate and contextually appropriate responses.

Our implementation uses:
- Groq for language model inference
- ChromaDB for vector storage and retrieval
- SentenceTransformer for generating text embeddings

## Features

- Document loading from a specified directory
- Text chunking for efficient processing
- Embedding generation using SentenceTransformer
- Vector storage and retrieval using ChromaDB
- Question answering using Groq's language model
- Persistent storage of document embeddings
- Customizable chunk size and overlap
- Flexible retrieval settings

## Project Structure

```
.
├── rag_implementation.py
├── .env
├── requirements.txt
├── README.md
├── LICENSE
├── news_articles/
│   ├── article1.txt
│   ├── article2.txt
│   └── ...
└── chroma_persistent_storage/
```

- `rag_implementation.py`: Main script containing the RAG implementation
- `.env`: Environment file for storing API keys
- `requirements.txt`: List of Python dependencies
- `news_articles/`: Directory containing text documents for processing
- `chroma_persistent_storage/`: Directory for ChromaDB's persistent storage

## Prerequisites

- Python 3.7+
- Groq API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Sarim-MBZUAI/mini-RAG_project.git
   cd mini-RAG_project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your documents:
   - Place your text documents in the `news_articles` directory.
   - Ensure all documents are in `.txt` format.

2. Run the script:
   ```
   python basic_rag.py
   ```

3. The script will process the documents and wait for user input. Enter your questions when prompted.

## How It Works

1. **Document Loading**: The script loads all `.txt` files from the `news_articles` directory.

2. **Text Chunking**: Each document is split into smaller chunks to facilitate processing and retrieval.

3. **Embedding Generation**: The SentenceTransformer model generates embeddings for each text chunk.

4. **Vector Storage**: Embeddings are stored in ChromaDB, creating a searchable vector space.

5. **Query Processing**: When a question is asked, it's converted into an embedding.

6. **Retrieval**: ChromaDB finds the most similar document chunks based on the query embedding.

7. **Context Preparation**: Retrieved chunks are combined to form a context.

8. **Response Generation**: The context and question are sent to Groq's language model, which generates a final answer.

## Customization

You can customize various aspects of the RAG system:

- **Chunk Size**: Modify the `chunk_size` parameter in `split_text()` to change the size of text chunks.
- **Chunk Overlap**: Adjust the `chunk_overlap` parameter to control how much chunks overlap.
- **Number of Retrieved Results**: Change `n_results` in `query_documents()` to retrieve more or fewer chunks.
- **Embedding Model**: Modify the `model_name` in `SentenceTransformerEmbeddingFunction` to use a different embedding model.
- **Language Model**: Change the `model` parameter in `generate_response()` to use a different Groq model.

## Advanced Usage

### Using Custom Embeddings

To use custom embeddings instead of SentenceTransformer:

1. Create a new class that inherits from `embedding_functions.EmbeddingFunction`.
2. Implement the `__call__` method to generate embeddings using your preferred method.
3. Replace `SentenceTransformerEmbeddingFunction` with your custom class.

### Implementing Batch Processing

For large document collections, implement batch processing:

1. Modify `load_documents_from_directory()` to yield batches of documents.
2. Process each batch separately in the main script.
3. Use ChromaDB's batch insertion capabilities for efficient storage.

<!-- ## Troubleshooting

- **API Key Issues**: Ensure your Groq API key is correctly set in the `.env` file.
- **Out of Memory Errors**: Reduce `chunk_size` or implement batch processing for large document collections.
- **Slow Performance**: Consider using a more powerful embedding model or optimize the chunk size.
- **Irrelevant Results**: Adjust the number of retrieved chunks or fine-tune the embedding model. -->

## Performance Considerations

- The choice of embedding model significantly impacts both speed and accuracy.
- Larger chunk sizes may improve context coherence but can slow down retrieval.
- Consider using GPU acceleration for embedding generation with large document collections.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.


