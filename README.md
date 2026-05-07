# NCERT RAG System - Local Setup Guide

A Retrieval-Augmented Generation (RAG) system for NCERT content using Chroma + Groq (LLM) and local embeddings.

## What You'll Need

- Python 3.10 or higher
- A Groq API key (`GROQ_API_KEY`)
- RTX 3060 GPU (or any CUDA-compatible GPU)
- NCERT content as text (recommended) or PDFs

## Quick Start Guide

### Step 1: Install Dependencies (uv)

```bash
uv sync
```

### Step 2: Set Groq API Key

```bash
export GROQ_API_KEY="your_groq_api_key"
```

### Step 3: Add Your NCERT Data

This repo is set up to ingest a single text file by default:

- Put all content into `Books/text.txt`

You can also ingest PDFs or a directory of text files by overriding the CLI `--pdf-dir` option.

```bash
# Default expected path:
ls Books/text.txt
```

### Step 4: Create Vector Database

Run the setup script to process your corpus:

```bash
uv run python setup_rag.py
```

This will:
- Load documents from `Books/text.txt` (by default)
- Split them into chunks
- Create embeddings using your GPU
- Store everything in a vector database (`vector_db/` folder)

**Note:** First run will download the embedding model (~90MB). This happens once.

### Step 5: Ask Questions!

Start the interactive Q&A system:

```bash
uv run python query_rag.py
```

### Optional: Streamlit UI

```bash
uv run streamlit run streamlit_code.py
```

Then type your questions! For example:
- "What is photosynthesis?"
- "Explain Newton's laws of motion"
- "What are the causes of the French Revolution?"

Type `quit` or `exit` to stop.

## Folder Structure

```
ncert-rag/
├── setup_rag.py          # Script to create vector database
├── query_rag.py          # Script to ask questions
├── requirements.txt      # Python dependencies
├── Books/                # Corpus text lives here (default)
└── vector_db/            # Generated vector database (created automatically)
```

## Recommended CLI (optional)

The refactor also adds a small CLI:

```bash
uv run python -m ncert_rag.cli ingest
uv run python -m ncert_rag.cli ask "What is photosynthesis?" --show-sources
```

## Configuration

### Change the Groq Model

Use the CLI flag:

```bash
uv run python -m ncert_rag.cli ask "What is photosynthesis?" --groq-model llama-3.3-70b-versatile
```

### Adjust Chunk Size

Edit `setup_rag.py` to change how text is split:

```python
chunk_size=1000,        # Make smaller for more precise retrieval
chunk_overlap=200,      # Increase for better context
```

### Number of Retrieved Documents

Use the CLI flag (higher can improve accuracy but may cost more/slow down):

```bash
uv run python -m ncert_rag.cli ask "Explain photosynthesis" --top-k 12
```

## Troubleshooting

### "CUDA out of memory"

1. Use smaller model: `llama3.2:3b` instead of `llama3.1:8b`
2. Or change device to CPU in `setup_rag.py`:
   ```python
   model_kwargs={'device': 'cpu'}
   ```

### "No module named..."

Make sure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

### "Vector database not found"

Run `python setup_rag.py` first to create the database.

## Tips for Better Results

1. **Start with 1-2 textbooks** - Test with smaller dataset first
2. **Ask specific questions** - "What is photosynthesis?" works better than "Tell me about plants"
3. **Check sources** - The system shows which pages it used to answer
4. **Re-run setup if you add new PDFs** - The database needs to be updated

## Next Steps

- Add more NCERT textbooks to expand knowledge base
- Experiment with different Ollama models
- Adjust chunk size and overlap for better retrieval
- Build a web interface using Streamlit or Gradio

## System Requirements

- **RAM:** 8GB minimum, 16GB recommended
- **GPU VRAM:** 6GB minimum (RTX 3060 has 12GB - perfect!)
- **Storage:** ~2-5GB for models and vector database

Happy learning! 📚
