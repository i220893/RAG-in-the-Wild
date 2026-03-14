from src.pipelines.rag_fusion import run_rag_fusion
from src.pipelines.hyde import run_hyde
from src.pipelines.crag import run_crag
from src.pipelines.graph_rag import run_graph_rag

PIPELINES = {
    "rag_fusion": run_rag_fusion,
    "hyde": run_hyde,
    "crag": run_crag,
    "graph_rag": run_graph_rag
}

def run_pipeline(pipeline_name: str, query: str, top_k: int = 5):
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINES[pipeline_name](query, top_k=top_k)
