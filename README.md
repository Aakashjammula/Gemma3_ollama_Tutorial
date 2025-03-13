# Gemma3 Tutorial

## Overview
This tutorial demonstrates how to use the new Gemma3 model for various generative AI tasks, including OCR (Optical Character Recognition) and RAG (Retrieval-Augmented Generation) in ollama. Gemma3 supports text and image inputs, over 140 languages, and a long 128K context window. 

### Supported Frameworks:
- **Hugging Face**: Use Gemma with the Transformers library.
- **NVIDIA Keras**: Finetune Gemma with LoRA and distributed training on TPUs.
- **Ollama**: Run Gemma models locally for inference.
- **JAX**: Fine-tune, shard, and optimize Gemma using JAX.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install gradio langchain_ollama langchain_chroma langchain_core langchain_community langchain_text_splitters pillow base64
```

## Project Files
- `gemma3_ocr.py` - Handles OCR and image-based AI chat.
- `gemma3_rag.py` - Implements RAG-based document querying from PDFs.

## Usage

### 1. RAG-based PDF Querying (`gemma3_rag.py`)
This script processes PDFs, stores them in a vector database, and allows users to query them using `gemma3:4b`.

#### Steps:
1. Run the script:
   ```bash
   python gemma3_rag.py
   ```
2. Upload PDFs.
3. Ask questions based on document content.

#### Features:
- PDF processing using `PyPDFLoader`.
- Vector search using `Chroma`.
- AI-powered question-answering with `gemma3:4b`.

### 2. OCR and AI Chat (`gemma3_ocr.py`)
This script allows users to chat with the AI using text and images.

#### Steps:
1. Run the script:
   ```bash
   python gemma3_ocr.py
   ```
2. Enter a message and optionally upload an image.
3. Get AI-generated responses based on text and visual inputs.

#### Features:
- Image-to-text conversion with base64 encoding.
- Chatting with AI using `gemma3:4b`.
- Gradio interface for easy interaction.

## Deployment
Both applications use Gradio for the user interface. The scripts can be deployed on local machines or cloud platforms.

## License
This project is open-source under the MIT License.
