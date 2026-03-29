from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from fastapi.responses import StreamingResponse

logger=get_logger(__name__)

app=FastAPI(title="RAG API")

pipeline=RAGPipeline()

class QueryRequest(BaseModel):
    query:str

@app.get("/")
def home():
    return {"message":"RAG API is running"}

@app.post("/ask")
def ask_ques(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")

        ans=pipeline.run(request.query)

        logger.info("Response generated successfully")

        return ans
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ask-stream")
def ask_stream(request: QueryRequest):
    try:
        logger.info(f"Streaming query: {request.query}")

        def generate():
            try:
                ans=pipeline.run(request.query)

                for word in ans.split():
                    yield word+" "

            except Exception as e:
                raise CustomException(str(e))
            
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"Streaming error : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))