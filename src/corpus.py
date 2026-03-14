import os
import yaml
import chromadb
from chromadb.utils import embedding_functions
from src.data_loader import load_examples, get_passages_for_retrieval
from pathlib import Path

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
        
        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="crag_corpus_1000",
            embedding_function=self.embedding_fn
        )

    def build_index(self, dataset_path=None, limit=None):
        """
        Extract all snippets from the dataset and add them to the collection.
        """
        path = dataset_path or CONFIG.get("dataset_path")
        print(f"Building index from {path}...")
        
        # Check if collection has data already
        if self.collection.count() > 0:
            print(f"Index already contains {self.collection.count()} items. Moving on.")
            return

        documents = []
        metadatas = []
        ids = []
        
        doc_set = set() # To avoid duplicate snippets
        
        count = 0
        for example in load_examples(path=path, limit=limit):
            for sr in example.get("search_results", []):
                snippet = sr.get("page_snippet")
                if snippet and snippet not in doc_set:
                    doc_set.add(snippet)
                    documents.append(snippet)
                    metadatas.append({
                        "page_name": sr.get("page_name", ""),
                        "page_url": sr.get("page_url", ""),
                        "interaction_id": example.get("interaction_id", "")
                    })
                    ids.append(f"doc_{count}")
                    count += 1
            
            if count % 100 == 0 and count > 0:
                 print(f"Processed {count} unique snippets...")

        # Add to chroma in batches if large
        batch_size = 5461 # Max batch size for chroma
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
            
        print(f"Finished building index with {len(documents)} snippets.")

    def retrieve(self, query: str, top_k: int = None):
        """
        Retrieve top-k snippets for a query.
        """
        k = top_k or CONFIG.get("top_k", 5)
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "score": results["distances"][0][i], # Note: distances, lower is better
                "metadata": results["metadatas"][0][i]
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
