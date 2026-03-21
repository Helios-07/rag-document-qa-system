import os
import yaml
import sys
import pickle

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_text
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.embedding.embedding import EmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Config file loaded successfully")
        return config

    except Exception as e:
        logger.error("Error loading config file")
        raise CustomException(e, sys)


def main():
    try:
        logger.info("Starting ingestion pipeline")

        config = load_config()

        data_dir = config["data"]["input_path"]
        chunk_size = config["chunking"]["chunk_size"]
        overlap = config["chunking"]["overlap"]

        all_chunks = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files = os.listdir(data_dir)

        if not files:
            logger.warning("No files found in data directory")

        for file in files:
            file_path = os.path.join(data_dir, file)

            try:
                logger.info(f"Processing file: {file}")

                text = load_document(file_path)

                if not text.strip():
                    logger.warning(f"Empty content in file: {file}")
                    continue

                chunks = chunk_text(text, chunk_size, overlap)

                all_chunks.extend(chunks)

                logger.info(f"Chunks created for {file}: {len(chunks)}")

            except Exception as e:
                logger.error(f"Skipping file {file} due to error: {str(e)}")
                continue

        logger.info(f"Total chunks from all files: {len(all_chunks)}")

        print(f"Total chunks from all files: {len(all_chunks)}")

        embedding_model=EmbeddingModel()
        embeddings=embedding_model.encode(all_chunks)

        logger.info(f"Embeddings shape: {embeddings.shape}")


        dimension=embeddings.shape[1]
        vector_store=FAISSVectorStore(dimension)
        vector_store.add_embeddings(embeddings)

        vector_store.save()

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/chunks.pkl", 'wb') as f:
            pickle.dump(all_chunks, f)
        logger.info("Chunks saved successfully")

        query="What do you mean by LR parsing?"

        query_embedding=embedding_model.encode([query])[0]

        distances,indices=vector_store.search(query_embedding, top_k=3)

        print("\n Query:", query)
        print("\nTop Results:\n")

        for i in indices[0]:
            print(all_chunks[i][:300])
            print("\n" + "-" * 50 + "\n")

    except Exception as e:
        logger.error("Pipeline execution failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()