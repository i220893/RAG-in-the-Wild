import os
import yaml
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_loader import load_examples
from langchain_core.documents import Document

# Load config
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        # Fallback to example if not found
        config_path = Path("config/config.example.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

class CorpusIndex:
    def __init__(self, chroma_path=None, embedding_model=None):
        self.chroma_path = chroma_path or CONFIG.get("chroma_path", "chroma_db")
        self.model_name = embedding_model or CONFIG.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Initialize embedding function
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Initialize chroma client
        self.vectorstore = Chroma(
            collection_name="crag_corpus_1000",
            embedding_function=self.embeddings,
            persist_directory=self.chroma_path
        )

    def build_index(self, dataset_path=None, limit=None):
        """
        Extract all snippets from the dataset and add them to the collection.
        """
        path = dataset_path or CONFIG.get("dataset_path")
        print(f"Building index from {path}...")
        
        # Check if collection has data already
        if self.vectorstore._collection.count() > 0:
            print(f"Index already contains {self.vectorstore._collection.count()} items. Moving on.")
            return

        documents = []
        doc_set = set() # To avoid duplicate snippets
        
        count = 0
        for example in load_examples(path=path, limit=limit):
            for sr in example.get("search_results", []):
                snippet = sr.get("page_snippet")
                if snippet and snippet not in doc_set:
                    doc_set.add(snippet)
                    
                    doc = Document(
                        page_content=snippet,
                        metadata={
                            "page_name": sr.get("page_name", ""),
                            "page_url": sr.get("page_url", ""),
                            "interaction_id": example.get("interaction_id", "")
                        }
                    )
                    documents.append(doc)
                    count += 1
            
            if count % 100 == 0 and count > 0:
                 print(f"Processed {count} unique snippets...")

        # Add to chroma
        if documents:
            self.vectorstore.add_documents(documents)
            
        print(f"Finished building index with {len(documents)} snippets.")

    def retrieve(self, query: str, top_k: int = None):
        """
        Retrieve top-k snippets for a query.
        """
        k = top_k or CONFIG.get("top_k", 5)
        # Using similarity_search_with_score to get distances
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results to match previous interface
        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "score": score, # In Chroma/LangChain, lower distance is better
                "metadata": doc.metadata
            })
        return formatted

def build_index(dataset_path: str = None, limit: int = None):
    index = CorpusIndex()
    index.build_index(dataset_path, limit)
    return index

def load_index():
    return CorpusIndex()

if __name__ == "__main__":
    # Test
    idx = build_index(limit=10) # Small limit for test
    results = idx.retrieve("Who is the CEO of Google?")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Text: {r['text'][:100]}...")
        print(f"Source: {r['metadata'].get('page_url')}")
        print("-" * 20)
