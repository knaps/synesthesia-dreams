import pandas as pd
import numpy as np
import logging
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OLD_SYN_CSV_PATH = 'synesthesia/synesthesia_processed.csv'
OLD_BASE_CSV_PATH = 'synesthesia/baseline_processed.csv'
NEW_SYN_PKL_PATH = 'synesthesia/synesthesia_processed.pkl'
NEW_BASE_PKL_PATH = 'synesthesia/baseline_processed.pkl'

# --- Helper Function (copied from synesthesia.py) ---

def parse_embedding(embedding_str):
    """Parses the string representation of an embedding into a numpy array."""
    if pd.isna(embedding_str):
        return None
    try:
        # Handle potential string formats from CSV:
        # 1. '{0.1, 0.2, ...}' or '[0.1, 0.2, ...]' -> remove brackets/braces
        # 2. '0.1 0.2 0.3 ...' (space separated, common from df.to_csv)
        cleaned_str = str(embedding_str).strip('{}[] ')

        # Revised parsing: Split by space and convert elements to float
        # This assumes the string looks like '[0.1 0.2 0.3 ...]' or similar after stripping brackets
        elements = cleaned_str.split()
        arr = np.array(elements, dtype=np.float32)
        # Check if parsing resulted in a non-empty array
        if arr.size == 0:
             raise ValueError("Parsing resulted in an empty array.")
        return arr

    except Exception as e:
        logging.warning(f"Could not parse embedding string: '{str(embedding_str)[:100]}...' Error: {e}")
        return None

# --- Conversion Function ---

def convert_csv_to_pickle(csv_path, pkl_path):
    """Reads a CSV cache file, parses embeddings, and saves as Pickle."""
    if not os.path.exists(csv_path):
        logging.warning(f"CSV file not found: {csv_path}. Skipping conversion.")
        return

    logging.info(f"Converting {csv_path} to {pkl_path}...")
    try:
        # Read CSV, ensuring date parsing
        df = pd.read_csv(csv_path, parse_dates=['created_utc'])
        logging.info(f"Read {len(df)} rows from {csv_path}.")

        # Parse the 'embedding' column using the corrected logic
        if 'embedding' in df.columns:
            logging.info("Parsing embeddings...")
            original_len = len(df)
            df['embedding'] = df['embedding'].apply(parse_embedding)
            df = df.dropna(subset=['embedding'])
            parsed_len = len(df)
            if parsed_len < original_len:
                logging.warning(f"Dropped {original_len - parsed_len} rows due to embedding parsing errors during conversion.")

            # Verify embedding dimension after parsing
            if not df.empty:
                first_embedding_len = len(df['embedding'].iloc[0])
                logging.info(f"Embeddings parsed. First embedding length: {first_embedding_len}")
                if first_embedding_len <= 1:
                     logging.warning(f"Parsed embeddings seem to have incorrect length ({first_embedding_len}). Check parse_embedding function.")
            else:
                logging.warning("DataFrame is empty after parsing embeddings.")


        else:
            logging.warning(f"'embedding' column not found in {csv_path}.")

        # Save as Pickle
        df.to_pickle(pkl_path)
        logging.info(f"Successfully saved DataFrame to {pkl_path}")

    except Exception as e:
        logging.error(f"Failed to convert {csv_path}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    logging.info("Starting CSV to Pickle cache conversion...")
    convert_csv_to_pickle(OLD_SYN_CSV_PATH, NEW_SYN_PKL_PATH)
    convert_csv_to_pickle(OLD_BASE_CSV_PATH, NEW_BASE_PKL_PATH)
    logging.info("Conversion process finished.")
