from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_text
from src.embedding.embedding import EmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
from fastapi.responses import StreamingResponse
import yaml
import os
import pickle

logger=get_logger(__name__)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

app=FastAPI(title="RAG API")

pipeline=RAGPipeline()

class QueryRequest(BaseModel):
    query:str

@app.get("/")
def home():
    return {"message":"RAG API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask-stream")
def stream(request: QueryRequest):
    try:
        logger.info(f"Streaming query: {request.query}")

        return StreamingResponse(pipeline.run(request.query),
                                 media_type="text/plain")

    
    except Exception as e:
        logger.error(f"Streaming error : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/upload")
def upload_file(file:UploadFile=File(...)):
    try:
        config=load_config()

        chunk_size=config['chunking']['chunk_size']
        overlap=config['chunking']['overlap']
        model_name=config['embedding']['model_name']
        faiss_path = config["paths"]["faiss_index"]
        chunks_path = config["paths"]["chunks"]

        file_path=f"temp_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        text=load_document(file_path)

        chunks=chunk_text(text,chunk_size,overlap)

        embedding_model=EmbeddingModel(model_name=model_name)
        embeddings=embedding_model.encode(chunks)
        

        if os.path.exists(faiss_path):
            logger.info("Loading existing FAISS index")

            vector_store=FAISSVectorStore()
            vector_store.load()
        else:
            logger.info("Creating new FAISS index")

            dimension=embeddings.shape[1]
            vector_store=FAISSVectorStore(dimension)

        vector_store.add_embeddings(embeddings)

        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        vector_store.save()


        if os.path.exists(chunks_path):
            logger.info("Loading existing chunks")

            with open(chunks_path, 'rb') as f:
                old_chunks=pickle.load(f)
        else:
            logger.info("No existing chunks found")
            old_chunks=[]

        all_chunks=old_chunks+chunks

        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)

        with open(chunks_path, 'wb') as f:
            pickle.dump(all_chunks,f)
        logger.info(f"Total chunks stored: {len(all_chunks)}")

        pipeline.retriever=Retriever()
        return {"message": "File processed successfully"}
    
    except Exception as e:
        import traceback

        print("UPLOAD ERROR:")
        print(traceback.format_exc())   
        logger.error(f"UPLOAD ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))