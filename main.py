import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI()

# Initialize the client with AIPipe's base URL
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.aipipe.org/v1"  # <--- This routes traffic to AIPipe
)

# --- 1. Pydantic Models for Input Validation ---

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

# --- 2. Pydantic Models for STRICT Output Schemas ---

class Entity(BaseModel):
    name: str
    type: str

class Relationship(BaseModel):
    source: str
    target: str
    relation: str

class ExtractionResponse(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    reasoning_path: List[str] = Field(default_factory=list)
    hops: int

class SummaryResponse(BaseModel):
    community_id: str
    summary: str

# --- 3. Hardened Endpoints ---

@app.post("/extract-graph", response_model=ExtractionResponse)
def extract_graph(req: ExtractRequest):
    try:
        prompt = f"""
        You are a strict Graph Extraction system.
        Extract entities (ONLY types: Person, Organization, Product, Framework) and relationships (ONLY types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED) from the text.
        
        Text: "{req.text}"
        
        If no relevant entities or relationships exist, return empty lists. Do not invent data.
        """
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractionResponse,
            temperature=0.0
        )
        
        return response.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph-query", response_model=QueryResponse)
def graph_query(req: GraphQueryRequest):
    try:
        prompt = f"""
        You are a strict Graph Reasoning system. Answer the question using ONLY the provided graph data. 
        Do not use outside knowledge. If the answer cannot be found in the graph, state "Information not present in graph".
        
        Graph Data: {json.dumps(req.graph)}
        Question: "{req.question}"
        
        Trace your exact path through the nodes to find the answer. Count the edges traversed as 'hops'.
        """
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=QueryResponse,
            temperature=0.0
        )
        
        return response.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/community-summary", response_model=SummaryResponse)
def community_summary(req: CommunityRequest):
    try:
        prompt = f"""
        You are a Graph Summarization system. Summarize the following sub-community in 1-2 concise sentences.
        Focus on how the entities are connected via the relationships.
        
        Entities: {json.dumps(req.entities)}
        Relationships: {json.dumps(req.relationships)}
        """
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=SummaryResponse,
            temperature=0.3 
        )
        
        parsed_response = response.choices[0].message.parsed
        parsed_response.community_id = req.community_id 
        
        return parsed_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
