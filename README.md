# RAG Pipeline Assignment - AI Engineering

A complete **Retrieval-Augmented Generation (RAG)** pipeline implementation using LangChain, FAISS, and OpenAI/Anthropic APIs.

## Overview

This project demonstrates a working RAG system that:
1. Loads and processes sample documents
2. Creates vector embeddings
3. Stores embeddings in a local FAISS vector database
4. Retrieves relevant context for queries
5. Generates accurate answers using LLM

## Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- OpenAI API key OR Anthropic API key

## Installation

### Option 1: Local Setup with Virtual Environment (Recommended)

This project uses a dedicated virtual environment to isolate dependencies.

```bash
# 1. Create virtual environment
python3 -m venv rag_venv

# 2. Activate virtual environment
# On macOS/Linux:
source rag_venv/bin/activate
# On Windows:
# rag_venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=rag_assignment --display-name="RAG Assignment (Python 3.9)"

# 5. Launch Jupyter Notebook
jupyter notebook rag_pipeline.ipynb
```

**Important**: When the notebook opens, select the kernel **"RAG Assignment (Python 3.9)"** from the Kernel menu to use the isolated environment.

### Option 2: Quick Install (Not Isolated)

```bash
# Install required packages directly
pip install langchain langchain-openai langchain-anthropic langchain-community langchain-text-splitters faiss-cpu jupyter ipykernel notebook

# Launch Jupyter
jupyter notebook rag_pipeline.ipynb
```

### Option 3: Google Colab

1. Upload `rag_pipeline.ipynb` to Google Colab
2. Run the first cell to install dependencies
3. Enter your API key when prompted

## Configuration

In the notebook, configure which LLM to use:

```python
USE_OPENAI = True  # Set to False to use Claude/Anthropic
```

### API Keys

You'll be prompted to enter your API key when running the notebook:
- **OpenAI**: Get key from https://platform.openai.com/api-keys
- **Anthropic**: Get key from https://console.anthropic.com/

**Note**: For production use, see `azure_optional/` folder for Azure Key Vault integration (optional).

## Project Structure

```
.
├── rag_venv/            # Virtual environment (isolated dependencies)
├── rag_pipeline.ipynb   # Main Jupyter notebook with complete RAG pipeline
├── requirements.txt     # Python package dependencies
├── .gitignore           # Git ignore file (excludes venv, API keys, etc.)
├── README.md            # This file
└── azure_optional/      # Optional Azure Key Vault integration files
```

## Notebook Sections

### 1. Install Dependencies
Installs all required libraries and verifies installation

### 2. Configure API Keys
Sets up authentication for OpenAI or Anthropic

### 3. Load Documents
Creates 6 sample documents covering:
- Artificial Intelligence basics
- Machine Learning
- Deep Learning
- Natural Language Processing
- RAG systems
- Vector databases

### 4. Split Documents
Uses `RecursiveCharacterTextSplitter` to chunk documents:
- Chunk size: 500 characters
- Chunk overlap: 50 characters

### 5. Create Embeddings & Vector Store
- Embeddings: OpenAI `text-embedding-ada-002`
- Vector DB: FAISS (local, no cloud required)

### 6. Retrieve Relevant Chunks
Performs similarity search to find top-k relevant chunks

### 7. Generate Answer
Uses OpenAI GPT-3.5 or Claude to generate context-aware answers

### 8. Display Results
Shows query, retrieved context, and final answer together

## Example Query

**Query**: "What is RAG and how does it work?"

The system will:
1. Convert query to vector embedding
2. Search FAISS for top 3 most similar chunks
3. Pass retrieved context to LLM
4. Generate comprehensive answer

## Screenshots for Assignment

The notebook produces clear output showing:
- ✓ Libraries installed successfully
- ✓ Documents loaded and split into chunks
- ✓ Embeddings created
- ✓ FAISS vector store built
- ✓ Query retrieval results with sources
- ✓ Final generated answer

## Key Features

- **No Cloud Dependencies**: Uses local FAISS storage
- **Flexible LLM**: Works with OpenAI or Anthropic
- **Isolated Environment**: Dedicated virtual environment and Jupyter kernel
- **Clear Structure**: Each cell has a specific purpose
- **Educational**: Includes detailed explanations and examples
- **Production-Ready**: Uses best practices for RAG systems

## Technologies Used

- **LangChain**: RAG orchestration framework
- **FAISS**: Vector similarity search (Facebook AI)
- **OpenAI Embeddings**: text-embedding-ada-002
- **LLM**: GPT-3.5-turbo or Claude-3-Sonnet
- **Python**: Core programming language

## Usage

1. Open the notebook in Jupyter or Colab
2. Run cells sequentially from top to bottom
3. Enter your API key when prompted
4. Observe the output at each step
5. Take screenshots for assignment submission

## Additional Examples

The notebook includes 3 additional example queries:
- "What are the three types of machine learning?"
- "Explain what vector databases are used for"
- "What is the difference between AI and deep learning?"

## Troubleshooting

**Wrong Kernel**: If the notebook doesn't find packages, ensure you selected "RAG Assignment (Python 3.9)" kernel
- Go to Kernel → Change Kernel → RAG Assignment (Python 3.9)

**Import Error**: Make sure virtual environment is activated and packages are installed
```bash
source rag_venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**API Key Error**: Verify your API key is valid and has credits

**FAISS Error on Windows**: Try installing via conda
```bash
conda install -c conda-forge faiss-cpu
```

**Kernel Not Found**: Re-register the kernel
```bash
source rag_venv/bin/activate
python -m ipykernel install --user --name=rag_assignment --display-name="RAG Assignment (Python 3.9)"
```

## Assignment Submission Checklist

- [ ] Run all cells successfully
- [ ] Screenshot: Libraries installed
- [ ] Screenshot: Documents loaded and split
- [ ] Screenshot: FAISS vector store created
- [ ] Screenshot: Retrieved chunks with sources
- [ ] Screenshot: Final generated answer
- [ ] Verify notebook runs end-to-end

## References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/)

## License

This project is for educational purposes as part of an AI Engineering assignment.
