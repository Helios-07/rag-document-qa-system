from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

logger=get_logger(__name__)

def chunk_text(text:str, chunk_size:int, overlap:int):
    try:
        logger.info("Starting text chunking")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        sentences=sent_tokenize(text)

        chunks=[]
        curr_chunk=[]
        curr_len=0

        for sentence in sentences:
            sentence_len=len(sentence.split())

            if curr_len+sentence_len>chunk_size:
                chunks.append(" ".join(curr_chunk))

                overlap_words=overlap
                overlap_chunk=[]
                temp_len=0

                for s in reversed(curr_chunk):
                    temp_len+=len(s.split())
                    overlap_chunk.insert(0,s)
                    if temp_len>=overlap_words:
                        break

                curr_chunk=overlap_chunk
                curr_len=sum(len(s.split()) for s in curr_chunk)

            curr_chunk.append(sentence)
            curr_len+=sentence_len

        if curr_chunk:
            chunks.append(" ".join(curr_chunk))

        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks
    
    except Exception as e:
        logger.error("Error during chunking")
        raise CustomException(e,sys)