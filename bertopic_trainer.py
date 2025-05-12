import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
import hdbscan # BERTopic often uses HDBSCAN for clustering
from typing import List, Dict, Optional # Add typing
import pandas as pd # Add pandas explicitly for type hints
import time # Add time for potential delays if needed
from tqdm import tqdm # Add tqdm for progress bars
import torch # Add torch import

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence SentenceTransformer info logs which can be noisy during loading
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
ENV_PATH = '.env'
MODEL_SAVE_DIR = 'bertopic_model/'
MODEL_NAME = 'dreams_sentence_model'
# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# --- Helper Functions ---

def load_dotenv_config():
    """Loads environment variables and returns the DB URL."""
    load_dotenv(dotenv_path=ENV_PATH)
    db_url = os.getenv("DB_CONN_STR")
    if not db_url:
        logging.error(f"DB_CONN_STR not found in {ENV_PATH}. Please ensure it's set.")
        raise ValueError("Database connection string not configured.")
    return db_url

def get_db_engine(db_url):
    """Creates and returns a SQLAlchemy engine."""
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            logging.info("Database connection successful.")
        return engine
    except Exception as e:
        logging.error(f"Failed to create database engine or connect: {e}")
        raise

def load_dream_data(engine):
    """Loads dream data from the database."""
    logging.info("Loading dream data from database (reddit.r_dreams_v3)...")
    query = text("""
        SELECT id, dream AS text
        FROM reddit.r_dreams_v3
        WHERE dream IS NOT NULL AND LENGTH(dream) > 0 AND LENGTH(dream) < 20000
    """)
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Loaded {len(df)} dream records.")
        return df
    except Exception as e:
        logging.error(f"Error loading dream data: {e}")
        raise

def split_into_sentences(text):
    """Splits text into sentences using NLTK."""
    try:
        return nltk.sent_tokenize(text)
    except Exception as e:
        logging.warning(f"Could not tokenize text: {text[:100]}... Error: {e}")
        return []

def train_bertopic_model(sentences, embedding_model):
    """Trains the BERTopic model on the provided sentences using a pre-loaded embedding model."""
    logging.info(f"Starting BERTopic training on {len(sentences)} sentences...")

    # 1. Embedding Model (Passed as argument)
    if embedding_model is None:
        logging.error("Embedding model was not provided to train_bertopic_model.")
        raise ValueError("Embedding model is required.")
    logging.info("Using pre-loaded SentenceTransformer model.")

    # 2. Dimensionality Reduction (UMAP defaults are usually good)
    # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # 3. Clustering (HDBSCAN defaults are often suitable)
    # hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')

    # 4. Tokenization and Weighting (TF-IDF)
    # Exclude common dream-related words and use min_df
    #vectorizer_model = CountVectorizer(stop_words=["dream", "dreams", "dreamt", "dreaming", "english"], min_df=10)
    vectorizer_model = CountVectorizer(min_df=10)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # Helps reduce impact of globally common words

    # 5. Representation (MMR for diverse topic words)
    representation_model = MaximalMarginalRelevance(diversity=0.4)

    # 6. Initialize BERTopic
    # We pass the components explicitly for clarity and control
    topic_model = BERTopic(
        embedding_model=embedding_model,    # Step 1 - Embeddings
        # umap_model=umap_model,            # Step 2 - Dimensionality Reduction (using default)
        # hdbscan_model=hdbscan_model,      # Step 3 - Clustering (using default)
        vectorizer_model=vectorizer_model,  # Step 4 - Tokenizer
        ctfidf_model=ctfidf_model,          # Step 4 - TF-IDF
        representation_model=representation_model, # Step 5 - Topic Representation
        language='english',
        calculate_probabilities=False, # Saves memory if probabilities aren't needed later
        verbose=True
    )

    logging.info("BERTopic model initialized. Starting fitting process (this may take a long time and consume significant RAM)...")
    # Fit the model
    try:
        topics, _ = topic_model.fit_transform(sentences) # We don't need probabilities here
        logging.info("BERTopic model training complete.")
        logging.info(f"Found {len(topic_model.get_topic_info()) - 1} topics (excluding outlier topic -1).") # -1 is the outlier topic
    except Exception as e:
        logging.error(f"Exception occurred during topic_model.fit_transform: {e}", exc_info=True)
        raise # Re-raise the exception after logging

    return topic_model

# --- Sentence Splitting Helper ---
# Keep this outside the class for potential general use or make it a static method
def split_text_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    if pd.isna(text):
        return []
    try:
        return nltk.sent_tokenize(str(text))
    except Exception as e:
        logging.warning(f"Could not tokenize text: {str(text)[:100]}... Error: {e}")
        return []


# --- Dream Topic Analyzer Class ---

class DreamTopicAnalyzer:
    """Encapsulates BERTopic model loading and analysis for dream data."""

    def __init__(self, model_path: str = os.path.join(MODEL_SAVE_DIR, MODEL_NAME), embedding_model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Loads the pre-trained BERTopic model and its associated embedding model.

        Args:
            model_path: Path to the saved BERTopic model directory.
            embedding_model_name: Name of the SentenceTransformer model used during BERTopic training.
            device: The device to load the embedding model onto ('cuda', 'mps', 'cpu', or None for auto-detect).
        """
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.embedding_model: Optional[SentenceTransformer] = None # Store embedding model used by BERTopic
        self.topic_model: Optional[BERTopic] = None
        self._load_models() # Load both models

    def _load_models(self):
        """Loads the BERTopic model from the specified path."""
        if not os.path.exists(self.model_path):
            logging.error(f"BERTopic model not found at: {self.model_path}. Please train the model first.")
            raise FileNotFoundError(f"BERTopic model not found: {self.model_path}")
        try:
            # 1. Load the embedding model first
            logging.info(f"Loading SentenceTransformer embedding model: {self.embedding_model_name} on device: {self.device}")
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
                logging.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                 logging.error(f"Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}")
                 # Decide if you want to raise here or allow BERTopic loading without it (which will likely fail later)
                 raise RuntimeError(f"Could not load embedding model: {e}") from e

            # 2. Load the BERTopic model
            # BERTopic.load() should ideally handle finding/loading the embedding model if saved correctly,
            # but explicitly passing the loaded one ensures consistency.
            logging.info(f"Loading pre-trained BERTopic model from: {self.model_path}...")
            # Try loading *without* explicitly passing the embedding model first,
            # as BERTopic >= 0.15 handles this better if saved with save_embedding_model=True
            try:
                 self.topic_model = BERTopic.load(self.model_path)
                 logging.info("BERTopic model loaded successfully (embedding model likely loaded internally).")
                 # Verify the embedding model is accessible if needed later
                 if hasattr(self.topic_model, 'embedding_model') and self.topic_model.embedding_model is not None:
                     logging.info("BERTopic's internal embedding model seems accessible.")
                     # Optionally assign it to self.embedding_model if needed elsewhere,
                     # but be cautious about potential conflicts if different models are expected.
                     # self.embedding_model = self.topic_model.embedding_model
                 else:
                     logging.warning("BERTopic loaded, but internal embedding model not readily accessible. Explicit loading might be needed if issues arise.")

            except TypeError as te: # Catch potential errors if load expects embedding_model
                 logging.warning(f"BERTopic.load failed without explicit embedding_model ({te}). Retrying with explicit model...")
                 if self.embedding_model:
                     self.topic_model = BERTopic.load(self.model_path, embedding_model=self.embedding_model)
                     logging.info("BERTopic model loaded successfully (with explicitly passed embedding model).")
                 else:
                     logging.error("Cannot load BERTopic model: Explicit embedding model was not loaded successfully earlier.")
                     raise RuntimeError("Failed to load BERTopic model due to missing embedding model.") from te
            except FileNotFoundError: # Keep specific error for model file
                 logging.error(f"BERTopic model file not found at: {self.model_path}. Please train the model first.")
                 raise
            except Exception as e:
                 logging.error(f"An unexpected error occurred during BERTopic.load: {e}")
                 raise # Re-raise other unexpected errors

        except FileNotFoundError: # Catch outer FileNotFoundError if path doesn't exist at all
             logging.error(f"BERTopic model path not found: {self.model_path}. Please train the model first.")
             raise
        except Exception as e:
            logging.error(f"Error loading BERTopic model or its components: {e}")
            raise

    def prepare_sentences_from_dataframe(self, df: pd.DataFrame, text_col: str = 'text', id_col: str = 'id', label_col: Optional[str] = None) -> pd.DataFrame:
        """Splits text from a DataFrame into sentences, keeping track of IDs and optional labels."""
        if self.topic_model is None:
             # The __init__ should raise if loading fails, so this indicates an object state issue.
             logging.error("prepare_sentences_from_dataframe called but topic_model is None. Initialization likely failed.")
             raise RuntimeError("BERTopic model is not loaded. Cannot prepare sentences.")

        if text_col not in df.columns or id_col not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{id_col}' and '{text_col}' columns.")
        if label_col and label_col not in df.columns:
             raise ValueError(f"Input DataFrame must contain '{label_col}' column if specified.")

        logging.info(f"Preprocessing text from '{text_col}': Splitting into sentences...")
        sentences_data = []
        required_cols = [id_col, text_col]
        if label_col:
            required_cols.append(label_col)

        # Use tqdm for progress indication
        for _, row in tqdm(df[required_cols].iterrows(), total=len(df), desc="Splitting sentences"):
            doc_id = row[id_col]
            text = row[text_col]
            label = row[label_col] if label_col else None

            sentences = split_text_into_sentences(text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    sentence_info = {
                        'doc_id': doc_id,
                        'sentence_idx': i,
                        'sentence_text': sentence
                    }
                    if label_col:
                        sentence_info[label_col] = label
                    sentences_data.append(sentence_info)

        if not sentences_data:
            logging.warning("No sentences generated from the input DataFrame.")
            return pd.DataFrame(columns=['doc_id', 'sentence_idx', 'sentence_text'] + ([label_col] if label_col else []))

        sentences_df = pd.DataFrame(sentences_data)
        logging.info(f"Generated {len(sentences_df)} sentences for BERTopic transform.")
        return sentences_df

    def infer_topics_for_sentences(self, sentences_df: pd.DataFrame, sentence_text_col: str = 'sentence_text') -> pd.DataFrame:
        """Infers topics for sentences using the loaded BERTopic model."""
        if self.topic_model is None:
            raise RuntimeError("BERTopic model is not loaded. Ensure DreamTopicAnalyzer was initialized correctly.")
        if sentence_text_col not in sentences_df.columns:
            raise ValueError(f"Sentences DataFrame must contain '{sentence_text_col}' column.")
        if sentences_df.empty:
            logging.warning("Input sentences DataFrame is empty. Returning empty DataFrame.")
            sentences_df['topic_id'] = pd.Series(dtype=int) # Add column even if empty
            return sentences_df

        logging.info("Inferring topics for sentences using BERTopic model...")
        sentences_list = sentences_df[sentence_text_col].tolist()
        try:
            # Check if the topic model has an embedding model associated *after* loading
            # If not, or to be safe, generate embeddings manually and pass them
            embeddings_to_use = None
            if hasattr(self.topic_model, 'embedding_model') and self.topic_model.embedding_model is not None:
                 logging.info("BERTopic model seems to have an internal embedding model. Attempting transform directly.")
                 # We can try without explicit embeddings first, but it failed before.
                 # Let's generate them explicitly to be robust.
                 # pass # Let BERTopic handle it internally (if it works)

            # Explicitly generate embeddings if the internal one is missing or unreliable
            if self.embedding_model:
                 logging.info("Generating sentence embeddings explicitly...")
                 embeddings_to_use = self.embedding_model.encode(sentences_list, show_progress_bar=True, device=self.device)
                 logging.info(f"Generated embeddings of shape: {embeddings_to_use.shape}")
            else:
                 # This case should ideally be prevented by checks in __init__
                 logging.error("Cannot generate embeddings: self.embedding_model is None.")
                 raise RuntimeError("Embedding model not available for topic inference.")

            # Use transform, passing the explicitly generated embeddings
            logging.info("Running BERTopic transform with explicit embeddings...")
            topics, _ = self.topic_model.transform(sentences_list, embeddings=embeddings_to_use)
            sentences_df['topic_id'] = topics
            logging.info("Topic inference complete.")

            # Filter out outlier topic (-1) as it's not informative for comparison
            n_before = len(sentences_df)
            sentences_df = sentences_df[sentences_df['topic_id'] != -1].copy()
            n_after = len(sentences_df)
            logging.info(f"Removed {n_before - n_after} sentences assigned to outlier topic (-1).")

            if sentences_df.empty:
                logging.warning("No non-outlier topics assigned to any sentences.")
            else:
                logging.info(f"Found {sentences_df['topic_id'].nunique()} unique non-outlier topics assigned.")

        except Exception as e:
            logging.error(f"Error during BERTopic transform: {e}")
            # Return the dataframe without the topic_id column in case of error
            return sentences_df.drop(columns=['topic_id'], errors='ignore')

        return sentences_df

    def aggregate_topics_to_dreams(self, sentences_with_topics_df: pd.DataFrame, original_df: pd.DataFrame, doc_id_col: str = 'doc_id', original_id_col: str = 'id', topic_col: str = 'topic_id') -> pd.DataFrame:
        """Aggregates sentence topics back to the dream level."""
        if self.topic_model is None:
            raise RuntimeError("BERTopic model is not loaded. Ensure DreamTopicAnalyzer was initialized correctly.")
        if doc_id_col not in sentences_with_topics_df.columns or topic_col not in sentences_with_topics_df.columns:
            raise ValueError(f"Sentences DataFrame must contain '{doc_id_col}' and '{topic_col}' columns.")
        if original_id_col not in original_df.columns:
             raise ValueError(f"Original DataFrame must contain '{original_id_col}' column.")

        logging.info("Aggregating sentence topics back to dream level...")
        # Group by doc_id and get a set of unique topics present in each dream
        dream_topics = sentences_with_topics_df.groupby(doc_id_col)[topic_col].apply(set).reset_index()

        # Ensure ID columns are compatible for merging (e.g., both strings)
        # Use .astype(str) defensively
        original_df_copy = original_df.copy() # Avoid modifying original DataFrame passed in
        original_df_copy[original_id_col] = original_df_copy[original_id_col].astype(str)
        dream_topics[doc_id_col] = dream_topics[doc_id_col].astype(str)

        # Merge back with the original df copy to get all original columns + the aggregated topics
        df_merged = pd.merge(original_df_copy, dream_topics, left_on=original_id_col, right_on=doc_id_col, how='left')

        # Fill NaN topic sets with empty sets for dreams that had no sentences or only outlier topics
        df_merged[topic_col] = df_merged[topic_col].apply(lambda x: x if isinstance(x, set) else set())

        # Drop the redundant doc_id column from the merge if it exists and differs from original_id_col
        if doc_id_col in df_merged.columns and doc_id_col != original_id_col:
            df_merged = df_merged.drop(columns=[doc_id_col])

        logging.info("Dream-level topic aggregation complete.")
        return df_merged

    def get_topic_info(self, filter_outliers: bool = True) -> pd.DataFrame:
        """Returns the topic info DataFrame from the loaded model."""
        if self.topic_model is None:
            raise RuntimeError("BERTopic model is not loaded. Ensure DreamTopicAnalyzer was initialized correctly.")
        topic_info_df = self.topic_model.get_topic_info()
        if filter_outliers:
            topic_info_df = topic_info_df[topic_info_df['Topic'] != -1].copy()
        return topic_info_df


def save_model(model: BERTopic, save_dir: str, model_name: str):
    """Saves the trained BERTopic model."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    logging.info(f"Saving BERTopic model to: {save_path}")
    try:
        # Use the recommended save method with embedding model saving enabled
        # serialization="safetensors" is generally preferred if dependencies allow
        model.save(save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
        logging.info("Model saved successfully.")
    except ImportError:
        logging.warning("safetensors not installed. Falling back to pickle serialization for BERTopic model.")
        model.save(save_path, serialization="pickle", save_ctfidf=True, save_embedding_model=True)
        logging.info("Model saved successfully using pickle.")
    except Exception as e:
        logging.error(f"Error saving BERTopic model: {e}")
        raise

# --- Main Execution ---
def main():
    """Main function to run the BERTopic training workflow."""
    logging.info("Starting BERTopic Training Workflow...")

    # 1. Initialize Embedding Model First (to test memory constraints)
    embedding_model = None # Initialize to None
    try:
        device = 'mps'
        logging.info(f"Attempting to initialize SentenceTransformer('all-MiniLM-L6-v2') on device: {device} BEFORE loading data...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        logging.info("Successfully initialized SentenceTransformer model.")
    except Exception as e:
        logging.error(f"Failed to initialize SentenceTransformer model even before loading data: {e}", exc_info=True)
        return # Exit if model loading fails even here

    # 2. Load Config and Connect to DB
    try:
        db_url = load_dotenv_config()
        engine = get_db_engine(db_url)
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return

    # 3. Load Data
    try:
        dreams_df = load_dream_data(engine)
        if dreams_df.empty:
            logging.error("No dream data loaded. Exiting.")
            return
    except Exception as e:
        logging.error(f"Failed during data loading: {e}")
        return

    # 3. Preprocess: Split into Sentences
    logging.info("Splitting dreams into sentences (using helper function)...")
    # Use the new class/helper to prepare sentences
    # Note: DreamTopicAnalyzer isn't strictly needed for *training*, but we can reuse the sentence prep
    # We don't instantiate the analyzer here as we are *training* a new model, not analyzing with a pre-trained one.
    sentences_data = []
    for _, row in tqdm(dreams_df.iterrows(), total=len(dreams_df), desc="Splitting sentences"):
        dream_id = row['id']
        text = row['text']
        sentences = split_text_into_sentences(text) # Use the helper
        for sentence in sentences:
            if sentence.strip():
                sentences_data.append({'dream_id': dream_id, 'sentence': sentence})

    if not sentences_data:
        logging.error("No sentences generated from dreams. Exiting.")
        return

    sentences_df = pd.DataFrame(sentences_data)
    logging.info(f"Generated {len(sentences_df)} sentences from {dreams_df['id'].nunique()} dreams.")


    # 5. Train Model (Pass the pre-loaded embedding model)
    try:
        # Pass only the sentence text and the pre-loaded model to the trainer
        topic_model = train_bertopic_model(sentences_df['sentence'].tolist(), embedding_model) # Pass the list of sentences
    except Exception as e:
        logging.error(f"Failed during BERTopic model training: {e}")
        return

    # 6. Save Model
    try:
        save_model(topic_model, MODEL_SAVE_DIR, MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed during model saving: {e}")
        return

    logging.info("BERTopic Training Workflow Completed Successfully.")

if __name__ == "__main__":
    main()
