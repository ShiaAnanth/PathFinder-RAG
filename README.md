# PathFinder@CISE
PathFinder@CISE is a Retrieval-Augmented Generation (RAG) system designed to help prospective students explore programs in the **College of Integrated Science and Engineering (CISE)** at James Madison University (JMU).

The system uses:
- A structured dataset of CISE majors, concentrations, and careers
- Semantic chunking and embeddings for retrieval
- A large language model to generate grounded, advisor-style answers based only on the provided data

## Key Features

- Uses a **JSON dataset** of CISE programs and concentrations (`CISE_programs.json`)
- Cleans and normalizes program descriptions and career text
- Splits long text into **semantic chunks** for better retrieval
- Creates dense embeddings using `sentence-transformers`
- Stores vectors in a **persistent ChromaDB** collection (`./vector_store`)
- Retrieves the most relevant chunks for a student’s question
- Sends those chunks + the question to a **Gemma** language model with strict system instructions
- Forces answers to stay grounded in the CISE dataset (no new majors or hallucinations)

## Repository Structure
```text
PathFinder-RAG/
│
├── app/
│   ├── app.py                 # Main RAG script (builds vector store + runs a sample query)
│   └── __init__.py
│
├── CISE_programs.json         # Cleaned program + concentration + career data
├── program_overviews.json     # Program-level overview text (optional / supporting)
├── PathFinder_CISE.ipynb      # Development notebook (exploration, testing, and analysis)
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignored files
└── README.md                  # Project documentation (this file)
```
## Requirements

## 📦 Requirements

This project relies on several Python libraries for web scraping, text processing, embeddings, vector storage, and model inference.

### Required Python Packages

The full dependency list is:

- `requests`
- `beautifulsoup4`
- `pandas`
- `lxml`
- `numpy`
- `scikit-learn`
- `transformers`
- `accelerate`
- `chromadb`
- `sentence-transformers`
- `bitsandbytes`  *(for model quantization and GPU/CPU efficiency)*

You can install all dependencies using:

```bash
pip install \
    requests \
    beautifulsoup4 \
    pandas \
    lxml \
    numpy \
    scikit-learn \
    transformers \
    accelerate \
    chromadb \
    sentence-transformers
```


## 🔧 Installation & Setup

### 1. Recommended Python Version
Use **Python 3.10 or Python 3.11**.

---

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```
To Activate it:

macOS / Linux: 
`source venv/bin/activate`
Windows: 
`venv\Scripts\activate`

3. Install all dependencies

4. If using Gemma locally (optional)

Gemma models may require a HuggingFace access token:

`export HF_TOKEN=your_token_here`

Also ensure you have:

A CUDA-enabled GPU

Correct PyTorch + bitsandbytes installation

## 🧾 Data: CISE Programs & Concentrations

The core dataset is stored in:

- `CISE_programs.json` – main structured data for:
  - Programs (e.g., ISAT, Computer Science, Engineering, etc.)
  - Concentrations or focus areas
  - Career descriptions where available

- `program_overviews.json` – additional overview descriptions for each program (if used).

Each program entry includes:
- `program`: the name of the major
- `description`: cleaned description text
- Optional `concentrations`: list of concentration objects with:
  - `name`
  - `description`
  - `careers`
- Optional `focus_areas`: similar structure for some programs

Before creating embeddings, the text is cleaned to:
- Remove non-breaking spaces (`\xa0`)
- Collapse extra whitespace
- Ensure consistent spacing for better model performance

## Semantic Chunking & Vector Store

Long descriptions are split into smaller chunks (~350 characters) using a simple sentence-based splitter:

- Text is split into sentences.
- Sentences are grouped together until a chunk reaches the target size.
- Each chunk is stored with metadata:
  - `id` (program + section)
  - `program` (which major it belongs to)
  - `type` (`overview`, `concentration`, or `focus_area`)
  - `text` (the actual chunk)

These chunks are then:

1. Embedded using `SentenceTransformer("all-MiniLM-L6-v2")`.
2. Stored in a **ChromaDB** persistent collection called `program_chunks` in `./vector_store`.

This allows fast semantic search over the program and concentration descriptions.

## RAG Pipeline: How a Question is Answered

When a student asks a question (for example:  
*“I like computers but not traditional Computer Science; are there majors with multiple concentrations that connect tech and society?”*), the system:

1. **Embeds the question** using the same `all-MiniLM-L6-v2` embedding model.
2. **Queries ChromaDB** to retrieve:
   - Program overviews (`type = "overview"`)
   - Matching concentrations (`type = "concentration"`)
3. **Combines the retrieved text** into a single context block.
4. **Builds a prompt** with:
   - A strict system instruction:
     - Only use information from the context
     - Do not invent new majors or rename existing ones
     - Do not treat concentrations as majors
     - Choose one best-fit program based on the data
   - The retrieved context
   - The student’s question
5. **Sends the prompt to the Gemma model** (e.g., `google/gemma-3-1b-it`) using the HuggingFace `transformers` API.
6. **Generates an answer** that:
   - Suggests a best-fit program (e.g., ISAT)
   - Uses concentrations and careers as supporting evidence
   - Stays fully grounded in the JSON data.

The system is designed to be **grounded and safe** for advising-style answers:

- It **can**:
  - Compare majors based on provided descriptions
  - Highlight relevant concentrations and career paths
  - Recommend the best-fit major for a given interest

- It **must not**:
  - Invent new majors
  - Rename or modify existing programs
  - Treat concentrations as standalone majors
  - Use information outside the JSON dataset

These rules are enforced directly in the system prompt given to the language model.

## Live Demo (Gradio)

A simplified demo version of the PathFinder@CISE app is available through Gradio on HuggingFace Spaces:
**Live App:** (https://shiaananth1-pathfinderdemo.hf.space/?logs=build&__theme=system&deep_link=GPkPZsXdb7I)

This version:
- Uses a lightweight model for fast inference
- Demonstrates the core RAG workflow
- Allows users to enter questions and see grounded responses


## This Project is Maintained by:

**Name:** Shia Ananth  
**Affiliation:** Integrated Science and Technology Major, College of Integrated Science and Engineering (CISE), James Madison University  

This repository was created as part of a course project on Retrieval-Augmented Generation and program advising tools.
