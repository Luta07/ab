import os
import json
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import AsyncOpenAI

app = FastAPI()

# Initialize the async client targeting AIPipe
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.aipipe.org/v1"
)

# --- Pydantic Models ---
class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict[str, Any]

class CommunityRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict[str, str]]

# --- Helper: Robust JSON Parser ---
def parse_llm_json(raw_text: str) -> dict:
    """Strips Markdown formatting if the LLM hallucinates a code block."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

# --- 0. Health Check ---
@app.get("/")
async def health_check():
    return {"status": "online", "engine": "GraphRAG Diagnostic Engine"}

# --- 1. Graph Extraction ---
@app.post("/extract-graph")
async def extract_graph(req: ExtractRequest):
    try:
        prompt = f"""
        Extract entities and relationships from the text. 
        Allowed Entity Types: Person, Organization, Product, Framework
        Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED, CREATED
        
        Text: "{req.text}"
        
        Provide response strictly in this JSON format:
        {{
          "entities": [{{ "name": "EntityName", "type": "Type" }}],
          "relationships": [{{ "source": "SourceName", "target": "TargetName", "relation": "RelationType" }}]
        }}
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        return parse_llm_json(response.choices[0].message.content)
    except Exception as e:
        # This will now print the exact API or Code error to your Render logs
        error_trace = traceback.format_exc()
        print(f"EXTRACTION ERROR:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

# --- 2. Graph Query ---
@app.post("/graph-query")
async def graph_query(req: GraphQueryRequest):
    try:
        prompt = f"""
        Answer the question using ONLY the provided graph data. Find the multi-hop reasoning path.
        
        Graph Data: {json.dumps(req.graph)}
        Question: "{req.question}"
        
        Provide response strictly in this JSON format:
        {{
          "answer": "Answer string",
          "reasoning_path": ["Node1", "Node2", "Node3"],
          "hops": 2
        }}
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        return parse_llm_json(response.choices[0].message.content)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"QUERY ERROR:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

# --- 3. Community Summarization ---
@app.post("/community-summary")
async def community_summary(req: CommunityRequest):
    try:
        prompt = f"""
        Summarize the relationships and entities of this specific sub-community in 1-2 sentences.
        
        Entities: {json.dumps(req.entities)}
        Relationships: {json.dumps(req.relationships)}
        
        Provide response strictly in this JSON format:
        {{
          "community_id": "{req.community_id}",
          "summary": "Summary string text here."
        }}
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = parse_llm_json(response.choices[0].message.content)
        result["community_id"] = req.community_id
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"SUMMARY ERROR:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
