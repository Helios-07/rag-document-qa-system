
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int):
    try:
        logger.info("Starting text chunking")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        
        text=text.replace("\n", " ")
        text=" ".join(text.split())

        sentences=sent_tokenize(text)

        if not sentences or len(sentences)<2:
            logger.warning("Sentence tokenization failed, using smart fallback chunking")

            words=text.split()
            chunks=[]
            step=chunk_size-overlap

            for i in range(0, len(words), step):
                chunk_words=words[i:i+chunk_size]
                chunk_text=" ".join(chunk_words)

                if "." in chunk_text:
                    last_period=chunk_text.rfind(".")
                    if last_period>100:
                        chunk_text=chunk_text[:last_period+1]

                chunks.append(chunk_text)

            logger.info(f"Total chunks created(fallback): {len(chunks)}")
            return chunks
        
        chunks=[]
        curr_chunk=[]
        curr_len=0

        for sentence in sentences:
            sentence_len=len(sentence.split())

            if curr_len+sentence_len>chunk_size:
                chunks.append(" ".join(curr_chunk))

                overlap_chunk=[]
                temp_len=0

                for s in reversed(curr_chunk):
                    temp_len+=len(s.split())
                    overlap_chunk.insert(0,s)
                    if temp_len<=overlap:
                        break

                curr_chunk=overlap_chunk
                curr_len=sum(len(s.split()) for s in curr_chunk)

            curr_chunk.append(sentence)
            curr_len+=sentence_len

        if curr_chunk:
            chunks.append(" ".join(curr_chunk))

        if not chunks:
            logger.warning("Chunking failed, forcing fallback chunk")
            return [text[:chunk_size * 5]]

        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks
    
    except Exception as e:
        logger.error("Error during chunking")
        raise CustomException(e,sys)
