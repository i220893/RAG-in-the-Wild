import { useState } from 'react';
import './App.css';

const PIPELINES = [
  { id: 'rag_fusion', name: 'RAG Fusion', desc: 'Query expansion + RRF' },
  { id: 'hyde', name: 'HyDE', desc: 'Hypothetical Document Embedding' },
  { id: 'crag', name: 'CRAG', desc: 'Corrective RAG with Citations' },
  { id: 'graph_rag', name: 'Graph RAG', desc: 'Graph-augmented retrieval' }
];

export default function App() {
  const [query, setQuery] = useState('');
  const [pipeline, setPipeline] = useState('rag_fusion');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          pipeline,
          top_k: 5
        }),
      });

      if (!response.ok) {
        throw new Error('Fallback to API error: ' + response.statusText);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>RAG in the Wild</h1>
        <p className="subtitle">Advanced Retrieval-Augmented Generation Case Study</p>
      </header>

      <div className="card">
        <div className="input-section">
          <div className="form-group">
            <label htmlFor="query">Research Query</label>
            <input
              id="query"
              type="text"
              placeholder="e.g. Which athlete won more Grand Slams, Federer or Nadal?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleRun()}
            />
          </div>

          <div className="form-group">
            <label htmlFor="pipeline">Retrieval Strategy</label>
            <select
              id="pipeline"
              value={pipeline}
              onChange={(e) => setPipeline(e.target.value)}
            >
              {PIPELINES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} — {p.desc}
                </option>
              ))}
            </select>
          </div>

          <button
            className="btn-primary"
            onClick={handleRun}
            disabled={loading || !query.trim()}
          >
            {loading ? <div className="loading-spinner" /> : 'Run Pipeline'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: '#ef4444', color: '#f87171' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="results-section">
          <div className="card answer-card">
            <label>Generated Answer ({result.pipeline})</label>
            <div className="answer-text">{result.answer}</div>
          </div>

          <div>
            <label style={{ marginBottom: '1rem', display: 'block' }}>Retrieved Context Chunks</label>
            <div className="chunks-grid">
              {result.context.map((chunk, i) => (
                <div key={i} className="chunk">
                  <div className="chunk-header">
                    <span className="score-badge">Score: {chunk.score.toFixed(4)}</span>
                    <a
                      href={chunk.metadata.page_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="source-link"
                    >
                      View Source
                    </a>
                  </div>
                  <div className="chunk-text">
                    {chunk.text.length > 300 ? chunk.text.slice(0, 300) + '...' : chunk.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
