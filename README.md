# 🧠 Local LLM-Powered PDF Q&A System

This project provides an **offline**, **private**, and **secure** way to interact with PDF documents using a local Large Language Model (LLM). It supports PDF parsing, intelligent chunking, semantic embedding via `sentence-transformers`, and fast retrieval using FAISS. All queries are processed locally using a Streamlit UI.

---

## 📋 Table of Contents

- [✨ Features](#-features)  
- [⚙️ Prerequisites](#️-prerequisites)  
- [🛠️ Installation Guide](#️-installation-guide)  
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)  
  - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)  
  - [Step 3: Install Dependencies](#step-3-install-dependencies)  
  - [Step 4: Download the Local LLM Model](#step-4-download-the-local-llm-model)  
- [📁 Project Structure](#-project-structure)  
- [🚀 How to Use](#-how-to-use)  
- [⚙️ Configuration Options](#️-configuration-options)  
- [🧰 Troubleshooting Common Issues](#-troubleshooting-common-issues)  

---

## ✨ Features

- ✅ **Offline Operation** — No internet or cloud dependency  
- 🔐 **Privacy-Focused** — Data never leaves your machine  
- 📄 **PDF Text Extraction** — Supports text-based PDFs  
- 🧩 **Smart Text Chunking** — Context-aware segmentation  
- 🔍 **Semantic Search** — Via FAISS for top-k retrieval  
- 🧠 **Local LLM Integration** — Runs using `llama-cpp-python`  
- 🌐 **Streamlit UI** — Simple and intuitive browser interface  
- 💾 **Session Management** — Keeps history and context  
- ⚙️ **Fully Configurable** — Chunk size, overlap, top-k, model path

---

## ⚙️ Prerequisites

- Python 3.9 or higher → [Download Python](https://www.python.org/)  
- Git → [Download Git](https://git-scm.com/)  
- Disk Space: 5–10 GB  
- RAM: 8 GB+ (for 7B quantized models like Mistral or LLaMA 2)

---

## 🛠️ Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### Step 2: Create and Activate a Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate    # On macOS/Linux
```

### Step 3: Install Dependencies
* Create a requirements.txt file with:

```requirements.txt
streamlit
PyMuPDF
langchain
langchain-community
sentence-transformers
faiss-cpu
llama-cpp-python
numpy
```

* Then install them:

```bash
pip install -r requirements.txt
```

### Step 4: Download the Local LLM Model
* Visit TheBloke on Hugging Face
* Download a .gguf file (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)
* Create a models/ directory and move the model file:

```bash
mkdir models
mv mistral-7b-instruct-v0.2.Q4_K_M.gguf models/
```
* Edit llm_utils.py to reflect the model path:

```python
LLM_MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", LLM_MODEL_FILENAME)
```

### 📁 Project Structure
```bash
your-project-name/
├── app.py                  # Streamlit App
├── pdf_utils.py            # PDF extraction logic
├── embed_utils.py          # Chunking + embedding
├── vector_store.py         # FAISS vector DB
├── llm_utils.py            # Local LLM inference
├── requirements.txt
├── venv/                   # Virtual environment
├── data/                   # PDF/chunk data
└── models/                 # Local LLM model (.gguf)
```

### 🚀 How to Use
1. Activate Environment

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Run App

```bash
streamlit run app.py
```

3. Interact

* Open browser at: http://localhost:8501
* Upload a PDF
* Ask questions — the LLM will respond contextually!

### ⚙️ Configuration Options

| Setting              | Location                    | Description                          |
| -------------------- | --------------------------- | ------------------------------------ |
| `chunk_size`         | `embed_utils.py`            | Max characters per chunk             |
| `chunk_overlap`      | `embed_utils.py`            | Overlap between chunks               |
| `k` (top-k)          | `app.py`, `vector_store.py` | # of top relevant chunks to retrieve |
| `LLM_MODEL_FILENAME` | `llm_utils.py`              | Filename of the local model          |

### 🧰 Troubleshooting Common Issues

❌ LLM model not found
* Check if the .gguf model is in the models/ folder
* Ensure the filename in llm_utils.py is correct

❌ No text extracted from PDF
* The file might be a scanned image PDF
* Use OCR (e.g., Tesseract OCR) to convert

❌ llama-cpp-python build fails
* Windows: Install Visual Studio Build Tools
* macOS/Linux: Ensure gcc, clang, or xcode is installed

❌ CUDA Out of Memory
* Use a smaller quantized model (e.g., Q4, Q2)
* Lower chunk_size and k settings

