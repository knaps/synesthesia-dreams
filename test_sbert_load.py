import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = 'all-MiniLM-L6-v2'
device = 'cpu' # Keep forcing CPU for now

try:
    logging.info(f"Attempting to initialize SentenceTransformer('{model_name}') on device: {device}...")
    # The library will print its own loading message here
    model = SentenceTransformer(model_name, device=device)
    logging.info(f"Successfully initialized SentenceTransformer model: {model_name}")
    # Optional: Test encoding a simple sentence
    logging.info("Attempting to encode a test sentence...")
    test_sentence = ["This is a test sentence."]
    embeddings = model.encode(test_sentence)
    logging.info(f"Successfully encoded test sentence. Embedding shape: {embeddings.shape}")

except Exception as e:
    logging.error(f"Failed during SentenceTransformer test: {e}", exc_info=True)

logging.info("Test script finished.")
