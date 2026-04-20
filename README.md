# RAG in the Wild — Case Study Assignment

This project implements and evaluates four advanced Retrieval-Augmented Generation (RAG) strategies—**RAG Fusion**, **HyDE**, **CRAG**, and **Graph RAG**—using a real-world-style corpus of web search results.

The goal is to analyze how different retrieval techniques handle noisy data and varied question types to determine the most reliable pipeline for a smart assistant.

## ✨ Key Features
- **Advanced RAG Pipelines**: Implementation of HyDE, RAG Fusion, Corrective RAG (CRAG), and Graph RAG.
- **Global Corpus Indexing**: Efficient vector search using ChromaDB.
- **Modern Full-Stack Architecture**: 
  - **Backend**: FastAPI with asynchronous query processing.
  - **Frontend**: Vite + React + Tailwind CSS for a premium, interactive chat experience.
- **Analytical Reporting**: A detailed recommendation report comparing the performance of each strategy.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- Node.js 18+
- [Git LFS](https://git-lfs.github.com/) (Recommended for large datasets)

### 2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/config.example.yaml config/config.yaml
```
Edit `config/config.yaml` to include your API keys (e.g., **Groq** or **Google Gemini**). 
> [!IMPORTANT]
> Do not use OpenAI API keys. This project is optimized for free/open-weight LLM providers.


---

## 📊 Dataset
This project utilizes the **CRAG Task 1 & 2 dev v4** dataset.
1. Download the dataset: [crag_task_1_and_2_dev_v4.jsonl.bz2](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2)
2. Decompress and place it in the `dataset/` directory.
3. Final path: `dataset/crag_task_1_and_2_dev_v4.jsonl`

The snippet texts from this dataset are used to build the global vector index.

---

## 🛠️ Usage

### Running the Evaluation
To run the automated benchmark across all implemented pipelines:
```bash
python run_evaluation.py
```

### Running the Application
**Start the Backend:**
```bash
python backend/app.py
```
*Port: 8000*

**Start the Frontend:**
```bash
cd frontend
npm install
npm run dev
```
*Port: 5173 (default Vite port)*

---

## 📂 Project Structure
```text
.
├── backend/            # FastAPI server
├── config/             # Configuration files
├── dataset/            # Local data storage (ignored by git)
├── docs/               # Documentation
├── frontend/           # React + Vite application
├── src/
│   ├── pipelines/      # RAG strategy implementations
│   ├── corpus.py       # Indexing and data loading
│   └── generation.py   # LLM interaction logic
├── run_evaluation.py   # Evaluation script
└── recommendation_report.md  # Final analysis
```

## 📜 Recommendation Summary
Based on our findings, **Corrective RAG (CRAG)** is the recommended strategy for production environments due to its self-correction mechanism and superior reliability with noisy web data. 

For detailed benchmarks, see [recommendation_report.md](./recommendation_report.md).
