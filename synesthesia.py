
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Add Random Forest
from sklearn.model_selection import GridSearchCV # Add GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA  # Optional
import shap  # Optional for SHAP analysis
import openai # Optional for GPT analysis
import logging
import argparse
import datetime # Add datetime import
import joblib # For caching grid search results
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from simpleaichat import AIChat
from pydantic import BaseModel, Field
from typing import List, Optional, Dict # Add Dict
import collections # Add collections import
from scipy.stats import chi2_contingency
from tqdm import tqdm
# BERTopic imports needed for grouping and type hints
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap # umap-learn library
import torch # For sentence transformer device selection
from bertopic_trainer import DreamTopicAnalyzer # Import directly as they are in the same directory


# --- Download NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence noisy HTTPX logs from simpleaichat/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
SYNESTHESIA_CSV_PATH = 'synesthesia/synesthesia_dreams.csv'
ENV_PATH = 'synesthesia/.env'
RANDOM_STATE = 42
N_SPLITS = 5 # For cross-validation
# Use pickle for caching arrays correctly
CACHED_SYN_PATH = 'synesthesia/synesthesia_processed.pkl'
CACHED_BASE_PATH = 'synesthesia/baseline_processed.pkl'
GRIDSEARCH_RESULTS_PATH = 'synesthesia/gridsearch_results.joblib' # Cache file for GridSearchCV
BERTOPIC_MODEL_PATH = 'synesthesia/bertopic_model/dreams_sentence_model' # Path to saved BERTopic model

# --- Helper Functions ---

def load_dotenv_config():
    """Loads environment variables and returns the DB URL."""
    load_dotenv(dotenv_path=ENV_PATH)
    db_url = os.getenv("DB_CONN_STR")
    if not db_url:
        logging.error(f"DB_CONN_STR not found in {ENV_PATH}. Please ensure it's set.")
        raise ValueError("Database connection string not configured.")
    openai_api_key = os.getenv("OPENAI_API_KEY") # Optional for GPT
    if not openai_api_key:
        logging.warning("OPENAI_API_KEY not found in .env. GPT features will be disabled.")
    return db_url, openai_api_key

def get_db_engine(db_url):
    """Creates and returns a SQLAlchemy engine."""
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as connection:
            logging.info("Database connection successful.")
        return engine
    except Exception as e:
        logging.error(f"Failed to create database engine or connect: {e}")
        raise

def parse_embedding(embedding_str):
    """Parses the string representation of an embedding into a numpy array."""
    if pd.isna(embedding_str):
        return None
    try:
        # Assuming format like '{0.1, 0.2, ...}' or '[0.1, 0.2, ...]'
        cleaned_str = embedding_str.strip('{}[] ')
        return np.fromstring(cleaned_str, sep=',', dtype=np.float32)
    except Exception as e:
        logging.warning(f"Could not parse embedding string: {embedding_str[:50]}... Error: {e}")
        return None

# --- Data Loading ---

def load_synesthesia_data(filepath=SYNESTHESIA_CSV_PATH):
    """Loads dream data for synesthesia users from a CSV file."""
    logging.info(f"Loading synesthesia data from {filepath}...")
    try:
        df = pd.read_csv(filepath, index_col=0)
        # Ensure 'embedding' column is parsed correctly if it's stored as a string
        if 'embedding' in df.columns and isinstance(df['embedding'].iloc[0], str):
             logging.info("Parsing string embeddings in synesthesia data...")
             df['embedding'] = df['embedding'].apply(parse_embedding)
             # Drop rows where embedding parsing failed
             original_len = len(df)
             df = df.dropna(subset=['embedding'])
             if len(df) < original_len:
                 logging.warning(f"Dropped {original_len - len(df)} rows due to embedding parsing errors.")

        # Rename columns for consistency if needed
        df = df.rename(columns={'dream': 'text', 'dream_id': 'id'}) # Adjust column names as necessary
        logging.info(f"Loaded {len(df)} synesthesia records.")
        return df[['id', 'author', 'text', 'created_utc', 'embedding']] # Select relevant columns
    except FileNotFoundError:
        logging.error(f"Synesthesia data file not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading synesthesia data: {e}")
        raise

def load_baseline_data_date_matched(engine, syn_df, min_length=50):
    """
    Loads baseline dream data from the database, matching the date distribution
    of the synesthesia dreams and excluding authors present in syn_df.
    """
    logging.info("Loading date-matched baseline dream data from database...")

    if 'created_utc' not in syn_df.columns:
        logging.error("Synesthesia DataFrame must have a 'created_utc' column.")
        raise ValueError("Missing 'created_utc' column in synesthesia data.")

    # Attempt to convert 'created_utc' to datetime, coercing errors to NaT
    original_len = len(syn_df)
    syn_df['created_utc'] = pd.to_datetime(syn_df['created_utc'], errors='coerce', utc=True) # Add utc=True to handle mixed timezones

    # Drop rows where conversion failed (NaT)
    syn_df = syn_df.dropna(subset=['created_utc'])
    if len(syn_df) < original_len:
       logging.warning(f"Dropped {original_len - len(syn_df)} synesthesia rows due to invalid 'created_utc' values.")

    if syn_df.empty:
       logging.error("Synesthesia DataFrame is empty after handling invalid dates.")
       raise ValueError("No valid 'created_utc' data found in synesthesia DataFrame.")

    logging.info("Successfully processed 'created_utc' in syn_df.")

    # Get daily counts and authors to exclude from the synesthesia dataframe
    # Now it's safe to use .dt accessor
    syn_df['date'] = syn_df['created_utc'].dt.date
    daily_targets = syn_df.groupby('date')['author'].agg(['count', set]).reset_index()
    target_authors = set(syn_df['author'].unique())
    logging.info(f"Calculated daily target counts for {len(daily_targets)} days.")

    baseline_results = []
    total_needed = daily_targets['count'].sum()
    total_fetched = 0

    # Base query structure
    base_query = """
    SELECT id, author, dream AS text, created_utc, embedding
    FROM reddit.r_dreams_v3
    WHERE embedding IS NOT NULL
    AND dream IS NOT NULL AND LENGTH(dream) >= :min_length
    AND author != '[deleted]'
    AND author NOT IN :exclude_authors
    AND DATE(created_utc) >= :start_date AND DATE(created_utc) <= :end_date
    ORDER BY RANDOM()
    LIMIT :limit
    """

    logging.info(f"Iterating through {len(daily_targets)} target dates to fetch baseline data...")
    # Wrap the iterator with tqdm for a progress bar
    for _, row in tqdm(daily_targets.iterrows(), total=len(daily_targets), desc="Fetching baseline data"):
        target_date = row['date']
        needed_count = row['count']
        exclude_authors_for_day = list(row['set'].union(target_authors)) # Exclude authors from this day AND all target authors

        # Try fetching from the exact date first
        start_date = target_date
        end_date = target_date
        params = {
            'min_length': min_length,
            'exclude_authors': tuple(exclude_authors_for_day), # Use tuple for IN clause
            'start_date': start_date,
            'end_date': end_date,
            'limit': needed_count
        }

        try:
            daily_df = pd.read_sql(text(base_query), engine, params=params)
            fetched_count = len(daily_df)

            # If not enough data, expand window to +/- 1 day
            if fetched_count < needed_count:
                logging.warning(f"Only found {fetched_count}/{needed_count} baseline records for {target_date}. Expanding window to +/- 1 day.")
                start_date = target_date - datetime.timedelta(days=1)
                end_date = target_date + datetime.timedelta(days=1)
                params['start_date'] = start_date
                params['end_date'] = end_date
                params['limit'] = needed_count # Still try to get the original needed count

                daily_df = pd.read_sql(text(base_query), engine, params=params)
                fetched_count = len(daily_df)

                # If still not enough, take what we have
                if fetched_count < needed_count:
                    logging.warning(f"Still only found {fetched_count}/{needed_count} baseline records for {target_date} (+/- 1 day). Using available records.")
                # Ensure we don't take more than needed if the expanded window has duplicates from other days already fetched
                daily_df = daily_df.sample(n=min(fetched_count, needed_count), random_state=RANDOM_STATE)


            baseline_results.append(daily_df)
            total_fetched += len(daily_df)
            logging.debug(f"Fetched {len(daily_df)} baseline records for target date {target_date}. Total fetched: {total_fetched}/{total_needed}")

        except Exception as e:
            logging.error(f"Error fetching baseline data for date {target_date}: {e}")
            # Decide whether to continue or raise the error
            # continue

    if not baseline_results:
        logging.error("No baseline data could be fetched.")
        return pd.DataFrame()

    # Concatenate all daily results
    df = pd.concat(baseline_results, ignore_index=True)

    # Drop potential duplicates introduced by overlapping date windows
    df = df.drop_duplicates(subset=['id'])
    logging.info(f"Total baseline records fetched after deduplication: {len(df)} (Target was {total_needed})")


    # Parse embeddings if necessary
    if not df.empty and isinstance(df['embedding'].iloc[0], str):
         logging.info("Parsing string embeddings in baseline data...")
         df['embedding'] = df['embedding'].apply(parse_embedding)
         # Drop rows where embedding parsing failed
         original_len = len(df)
         df = df.dropna(subset=['embedding'])
         if len(df) < original_len:
             logging.warning(f"Dropped {original_len - len(df)} baseline rows due to embedding parsing errors.")
    elif not df.empty and isinstance(df['embedding'].iloc[0], (list, np.ndarray)):
         # Ensure they are numpy arrays
         df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32) if x is not None else None)
         df = df.dropna(subset=['embedding']) # Also drop if conversion results in None

    logging.info(f"Finished loading and parsing baseline data. Final count: {len(df)}")
    return df


# --- Data Preparation ---

def prepare_data(syn_df, base_df):
    """Combines datasets, adds target label, handles missing embeddings."""
    logging.info("Preparing combined dataset...")
    syn_df['is_synesthesia'] = True
    base_df['is_synesthesia'] = False

    combined_df = pd.concat([syn_df, base_df], ignore_index=True)

    # Drop rows with missing embeddings or text
    original_len = len(combined_df)
    combined_df = combined_df.dropna(subset=['embedding', 'text'])
    if len(combined_df) < original_len:
        logging.warning(f"Dropped {original_len - len(combined_df)} rows due to missing embeddings or text.")

    # Ensure embeddings are numpy arrays
    combined_df['embedding'] = combined_df['embedding'].apply(lambda x: np.array(x, dtype=np.float32) if x is not None else None)

    # Check embedding dimensions
    first_embedding_len = len(combined_df['embedding'].iloc[0]) if not combined_df.empty else 0
    if first_embedding_len == 0:
         logging.error("No valid embeddings found after preparation.")
         raise ValueError("Embeddings are missing or invalid.")

    logging.info(f"Checking embedding consistency (first embedding length: {first_embedding_len})...")
    combined_df['embedding_len'] = combined_df['embedding'].apply(len)
    inconsistent = combined_df[combined_df['embedding_len'] != first_embedding_len]
    if not inconsistent.empty:
        logging.warning(f"Found {len(inconsistent)} embeddings with inconsistent lengths. Dropping them.")
        combined_df = combined_df[combined_df['embedding_len'] == first_embedding_len]

    combined_df = combined_df.drop(columns=['embedding_len'])

    if combined_df.empty:
        logging.error("Dataset is empty after cleaning inconsistent embeddings.")
        raise ValueError("No consistent embeddings found.")

    logging.info(f"Prepared dataset with {len(combined_df)} records.")
    logging.info(f"Class distribution:\n{combined_df['is_synesthesia'].value_counts(normalize=True)}")

    X = np.vstack(combined_df['embedding'].values)
    y = combined_df['is_synesthesia'].values.astype(int) # Use integer labels

    return combined_df, X, y

# --- Model Training & Evaluation ---

def build_pipeline(use_pca=False, n_components=100, C=1.0, penalty='l2', solver='liblinear', class_weight=None):
    """Builds a scikit-learn pipeline for classification."""
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)))
    # Classifier step will be defined in the grid search parameters
    # This function now primarily sets up the preprocessing steps
    # We add a placeholder 'clf' step name which GridSearchCV will populate
    steps.append(('clf', None)) # Placeholder for classifier
    return Pipeline(steps)

def evaluate_model_cv(pipeline, X, y, n_splits=N_SPLITS):
    """Evaluates the model using stratified cross-validation."""
    logging.info(f"Evaluating model using {n_splits}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    roc_auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    logging.info(f"Cross-Validation F1 (weighted): {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    logging.info(f"Cross-Validation ROC AUC: {np.mean(roc_auc_scores):.4f} +/- {np.std(roc_auc_scores):.4f}")
    return {'cv_f1_mean': np.mean(f1_scores), 'cv_roc_auc_mean': np.mean(roc_auc_scores)}

def train_and_evaluate_final(pipeline, X, y, test_size=0.2):
     """Trains the model on a train split and evaluates on a test split."""
     logging.info(f"Training final model on {1-test_size:.0%} of data and evaluating on {test_size:.0%}...")
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
     pipeline.fit(X_train, y_train)
     y_pred = pipeline.predict(X_test)
     y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

     f1 = f1_score(y_test, y_pred, average='weighted')
     roc_auc = roc_auc_score(y_test, y_pred_proba)

     logging.info(f"Test Set F1 (weighted): {f1:.4f}")
     logging.info(f"Test Set ROC AUC: {roc_auc:.4f}")
     logging.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred))
     logging.info("Confusion Matrix (Test Set):\n" + str(confusion_matrix(y_test, y_pred)))

     return pipeline, {'test_f1': f1, 'test_roc_auc': roc_auc}


def find_best_model(X, y, n_splits=N_SPLITS):
    """
    Performs GridSearchCV to find the best model and hyperparameters.

    Tests Logistic Regression and Random Forest, with and without PCA.
    Caches the best found pipeline to avoid re-running.
    """
    # --- Check for Cached Results ---
    if os.path.exists(GRIDSEARCH_RESULTS_PATH):
        logging.info(f"Loading best pipeline from cache: {GRIDSEARCH_RESULTS_PATH}")
        try:
            best_pipeline = joblib.load(GRIDSEARCH_RESULTS_PATH)
            logging.info(f"Successfully loaded cached pipeline: {best_pipeline}")
            # Note: We don't have the best score/params easily available here without saving more data,
            # but we have the best estimator itself, which is the primary goal.
            return best_pipeline
        except Exception as e:
            logging.error(f"Failed to load pipeline from cache: {e}. Re-running GridSearchCV.")

    logging.info("No valid cache found. Starting GridSearchCV to find the best model and hyperparameters...")

    # --- Define Pipelines and Grids (if cache doesn't exist) ---
    pipe_base = Pipeline([('scaler', StandardScaler()), ('clf', None)]) # clf is placeholder
    pipe_pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA(random_state=RANDOM_STATE)), ('clf', None)])

    # Define parameter grids
    param_grid_lr = {
        'clf': [LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')],
        'clf__C': [0.001, 0.01, 0.1, 1],
        'clf__penalty': ['l1', 'l2']
    }

    param_grid_rf = {
        'clf': [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')],
        'clf__n_estimators': [100, 200, 500],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
        # Add other RF params if needed: 'clf__min_samples_leaf': [1, 3]
    }

    # Parameter grid specific to PCA step (only used with pipe_pca)
    param_grid_pca = {
        'pca__n_components': [50, 100, 150, 200] # Components to test for PCA
    }

    # Combine grids for PCA pipelines
    param_grid_lr_pca = {**param_grid_lr, **param_grid_pca} # Combine LR grid with PCA grid
    param_grid_rf_pca = {**param_grid_rf, **param_grid_pca} # Combine RF grid with PCA grid


    # --- Grid Search Execution ---
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = 'roc_auc' # Use ROC AUC for scoring
    best_score = -1
    best_pipeline = None

    search_configs = [
        ("Logistic Regression (No PCA)", pipe_base, param_grid_lr),
        ("Random Forest (No PCA)", pipe_base, param_grid_rf),
        ("Logistic Regression (with PCA)", pipe_pca, param_grid_lr_pca),
        ("Random Forest (with PCA)", pipe_pca, param_grid_rf_pca),
    ]

    for name, pipe, params in search_configs:
        logging.info(f"--- Running GridSearchCV for: {name} ---")
        search = GridSearchCV(pipe, params, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        try:
            search.fit(X, y)
            logging.info(f"Best score ({scoring}) for {name}: {search.best_score_:.4f}")
            logging.info(f"Best params for {name}: {search.best_params_}")

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_pipeline = search.best_estimator_ # Get the best pipeline object
                logging.info(f"*** New overall best model found: {name} (Score: {best_score:.4f}) ***")

        except Exception as e:
            logging.error(f"GridSearchCV failed for {name}: {e}")


    if best_pipeline is None:
        logging.error("GridSearchCV failed to find any valid model.")
        raise RuntimeError("Model optimization failed.")

    logging.info(f"--- GridSearchCV Finished ---")
    logging.info(f"Overall best pipeline score ({scoring}): {best_score:.4f}")
    logging.info(f"Overall best pipeline configuration: {best_pipeline}")

    # --- Save the Best Pipeline to Cache ---
    try:
        logging.info(f"Saving the best pipeline to cache: {GRIDSEARCH_RESULTS_PATH}")
        joblib.dump(best_pipeline, GRIDSEARCH_RESULTS_PATH)
    except Exception as e:
        logging.error(f"Failed to save best pipeline to cache: {e}")

    return best_pipeline


# --- Model Introspection ---

def analyze_model_scores(df, pipeline, X):
    """Adds prediction scores and deciles to the DataFrame."""
    logging.info("Analyzing model scores...")
    df['score'] = pipeline.predict_proba(X)[:, 1] # Probability of being True (is_synesthesia)
    try:
        df['score_decile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
    except ValueError as e:
         logging.warning(f"Could not create 10 deciles due to non-unique score values: {e}. Adjusting bins.")
         # Fallback or alternative binning strategy
         df['score_decile'] = pd.qcut(df['score'], min(5, len(df['score'].unique())-1), labels=False, duplicates='drop')


    # Analyze prevalence per decile
    decile_analysis = df.groupby('score_decile')['is_synesthesia'].agg(['mean', 'count'])
    logging.info("Prevalence of 'is_synesthesia' per score decile:\n" + str(decile_analysis))

    # Plotting (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        decile_analysis['mean'].plot(kind='bar')
        plt.title('Prevalence of Synesthesia Dreams by Model Score Decile')
        plt.xlabel('Score Decile (0=Lowest, 9=Highest)')
        plt.ylabel('Proportion of Synesthesia Dreams')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('synesthesia_decile_analysis.png')
        logging.info("Saved decile analysis plot to synesthesia_decile_analysis.png")
        plt.close()
    except ImportError:
        logging.warning("Matplotlib not installed. Skipping decile plot generation.")

    return df, decile_analysis

def run_shap_analysis(pipeline, X_train, X_test, df_test):
    """ Performs SHAP analysis (example for Logistic Regression). """
    # Note: SHAP works best with tree models or requires specific explainers for others.
    # This is a basic example assuming the classifier step is named 'clf'.
    logging.info("Running SHAP analysis (example for LR)...")
    try:
        clf_step = pipeline.named_steps['clf']
        scaler_step = pipeline.named_steps['scaler'] # Need scaler for background data

        # Use a sample of the training data as background for SHAP
        # SHAP recommends using a representative background dataset (e.g., k-means centroids or a sample)
        background_data = shap.sample(scaler_step.transform(X_train), 100) # Scale background data

        # Create SHAP explainer for the classifier model
        # For Linear models like Logistic Regression
        explainer = shap.LinearExplainer(clf_step, background_data)

        # Calculate SHAP values for the test set (scaled)
        shap_values = explainer.shap_values(scaler_step.transform(X_test))

        # Summarize SHAP values (e.g., mean absolute value per feature)
        # Feature names would ideally correspond to embedding dimensions or PCA components
        feature_names = [f"emb_{i}" for i in range(X_test.shape[1])]
        if 'pca' in pipeline.named_steps:
             feature_names = [f"pc_{i}" for i in range(pipeline.named_steps['pca'].n_components_)]

        shap.summary_plot(shap_values, features=scaler_step.transform(X_test), feature_names=feature_names, show=False)

        # Save the plot (requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.savefig("synesthesia_shap_summary.png")
            logging.info("Saved SHAP summary plot to synesthesia_shap_summary.png")
            plt.close()
        except ImportError:
            logging.warning("Matplotlib not installed. Skipping SHAP plot saving.")

        # You can further analyze shap_values here, e.g., find top contributing features
        # Example: Get mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
        logging.info("Top 10 features by mean absolute SHAP value:\n" + str(feature_importance.head(10)))

        return shap_values, feature_importance

    except ImportError:
        logging.warning("SHAP library not installed. Skipping SHAP analysis.")
        return None, None
    except Exception as e:
        logging.error(f"Error during SHAP analysis: {e}")
        return None, None


# --- TF-IDF Introspection ---

def analyze_tfidf_deciles(df):
    """ Performs TF-IDF analysis on top vs bottom deciles and asks LLM for themes. """
    logging.info("Running TF-IDF analysis on top/bottom deciles...")
    if 'score_decile' not in df.columns:
        logging.error("DataFrame must have 'score_decile' column for TF-IDF analysis.")
        return

    try:
        # Preprocessing
        stop_words = set(stopwords.words('english'))
        # Ensure text column exists and handle potential NaN values
        if 'text' not in df.columns:
             logging.error("DataFrame must have 'text' column.")
             return
        df['processed_text'] = df['text'].fillna('').apply(
            lambda x: ' '.join([word.lower() for word in word_tokenize(str(x)) if word.isalpha() and word.lower() not in stop_words])
        )

        # Fit TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000) # Limit features for performance
        tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
        feature_names = vectorizer.get_feature_names_out()

        # Get indices for top and bottom deciles
        bottom_decile_indices = df[df['score_decile'] == 0].index
        top_decile_indices = df[df['score_decile'] == df['score_decile'].max()].index # Use max() in case of fewer than 10 deciles

        if len(bottom_decile_indices) == 0 or len(top_decile_indices) == 0:
            logging.warning("Not enough data in top or bottom deciles for TF-IDF analysis.")
            return

        # Calculate mean TF-IDF for each decile
        bottom_tfidf_mean = np.array(tfidf_matrix[bottom_decile_indices].mean(axis=0)).flatten()
        top_tfidf_mean = np.array(tfidf_matrix[top_decile_indices].mean(axis=0)).flatten()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'feature': feature_names,
            'bottom_mean_tfidf': bottom_tfidf_mean,
            'top_mean_tfidf': top_tfidf_mean
        })
        comparison_df['diff_top_bottom'] = comparison_df['top_mean_tfidf'] - comparison_df['bottom_mean_tfidf']

        # Get top N differentiating words
        n_words = 50
        top_words = comparison_df.sort_values('diff_top_bottom', ascending=False).head(n_words)['feature'].tolist()
        bottom_words = comparison_df.sort_values('diff_top_bottom', ascending=True).head(n_words)['feature'].tolist()

        logging.info(f"Top {n_words} words associated with HIGH scores (Synesthesia):\n{', '.join(top_words)}")
        logging.info(f"Top {n_words} words associated with LOW scores (Baseline):\n{', '.join(bottom_words)}")

        # LLM Analysis for Theming
        try:
            ai = AIChat(console=False, model='gpt-4o', api_key=os.getenv("OPENAI_API_KEY"))
            prompt_top = f"""The following words are most strongly associated with dreams from subreddit A (compared to subreddit B). Please group these words into meaningful semantic themes (3-5 themes usually works well). Provide only the theme names and the words belonging to each theme.

Words: {', '.join(top_words)}"""
            response_top = ai(prompt_top)
            logging.info(f"LLM Theming for HIGH score words:\n{response_top}")

            prompt_bottom = f"""The following words are most strongly associated with dreams from subreddit B (compared to subreddit A). Please group these words into meaningful semantic themes (3-5 themes usually works well). Provide only the theme names and the words belonging to each theme.

Words: {', '.join(bottom_words)}"""
            response_bottom = ai(prompt_bottom)
            logging.info(f"LLM Theming for LOW score words:\n{response_bottom}")

        except ImportError:
            logging.warning("simpleaichat not installed. Skipping LLM theming for TF-IDF words.")
        except Exception as e:
            logging.error(f"Error during LLM theming for TF-IDF words: {e}")

    except Exception as e:
        logging.error(f"Error during TF-IDF analysis: {e}")


# --- GPT Theme Tagging Introspection ---

# Pydantic models for batch processing
class TaggedDream(BaseModel):
    """Represents a single dream tagged with themes."""
    dream_id: str = Field(description="The unique identifier of the dream.")
    themes: List[str] = Field(description="List of applicable themes from the provided list.")

class TaggedDreamBatch(BaseModel):
    """Structured output for a batch of tagged dreams."""
    tagged_dreams: List[TaggedDream] = Field(description="A list of dreams tagged with their respective themes.")


# Default themes list - can be overridden
DEFAULT_THEMES_LIST = [
    'Sensory Perception (Visual, Auditory, Tactile)', 'Color', 'Emotion (Positive)', 'Emotion (Negative/Fear/Anxiety)',
    'Social Interaction (Family, Friends, Strangers)', 'Work/School/Professional Life', 'Movement/Action',
    'Spatial Environment (Indoors, Outdoors, Specific Locations)', 'Body Experience/Physicality',
    'Cognitive Processes (Thinking, Knowing, Remembering)', 'Surprise/Weirdness/Confusion', 'Lucidity/Dream Awareness',
    'Conflict/Chase/Escape', 'Transformation/Change', 'Abstract/Symbolic'
]

def analyze_gpt_themes(df, api_key, themes_list: List[str], sample_size=None):
    """ Tags dreams in top/bottom deciles using GPT against a predefined list and analyzes theme prevalence. """
    logging.info("Running GPT Per-Dream Theme Tagging analysis on top/bottom deciles...")
    if not api_key:
        logging.error("OpenAI API key not provided. Skipping GPT Theme Tagging.")
        return
    if 'score_decile' not in df.columns or 'text' not in df.columns:
        logging.error("DataFrame must have 'score_decile' and 'text' columns.")
        return

    try:
        from simpleaichat import AIChat
    except ImportError:
        logging.error("simpleaichat library not installed. Skipping GPT Theme Tagging.")
        return

    if not themes_list:
        logging.error("Themes list cannot be empty for per-dream tagging.")
        return

    logging.info(f"Using themes: {themes_list}")

    # Filter for top and bottom deciles
    bottom_decile = df[df['score_decile'] == 0].copy()
    top_decile = df[df['score_decile'] == df['score_decile'].max()].copy()

    if bottom_decile.empty or top_decile.empty:
        logging.warning("Not enough data in top or bottom deciles for GPT theme analysis.")
        return

    # Optional sampling to reduce cost/time
    if sample_size:
        logging.info(f"Sampling {sample_size} dreams from top and bottom deciles for GPT tagging.")
        if len(bottom_decile) > sample_size:
            bottom_decile = bottom_decile.sample(n=sample_size, random_state=RANDOM_STATE)
        if len(top_decile) > sample_size:
            top_decile = top_decile.sample(n=sample_size, random_state=RANDOM_STATE)

    df_to_tag = pd.concat([bottom_decile, top_decile]).copy() # Use .copy() to avoid SettingWithCopyWarning
    total_dreams_to_tag = len(df_to_tag)
    logging.info(f"Tagging {total_dreams_to_tag} dreams with GPT in batches...")

    ai = AIChat(console=False, model='gpt-4o', api_key=api_key)
    all_tagged_themes = {} # Dictionary to store {dream_id: themes_list}
    batch_size = 20 # Process 20 dreams per API call

    # Ensure 'id' column exists and is suitable for use as dream_id
    if 'id' not in df_to_tag.columns:
        logging.error("DataFrame must have an 'id' column to use as dream_id for batch tagging.")
        return
    # Convert id to string just in case it's numeric
    df_to_tag['id'] = df_to_tag['id'].astype(str)


    for i in tqdm(range(0, total_dreams_to_tag, batch_size), desc="Tagging dream batches"):
        batch_df = df_to_tag.iloc[i:min(i + batch_size, total_dreams_to_tag)]

        if batch_df.empty:
            continue

        # Construct the batch prompt
        batch_prompt_parts = []
        for _, row in batch_df.iterrows():
            # Ensure text is not excessively long for the prompt
            dream_text_truncated = str(row['text'])[:2000] # Truncate long dreams if necessary
            batch_prompt_parts.append(f"--- Dream ID: {row['id']} ---\n{dream_text_truncated}\n")

        batch_dream_text = "\n".join(batch_prompt_parts)

        prompt = f"""You will be given a batch of dream reports below, each marked with '--- Dream ID: [id] ---'.
For each dream report, assign the most relevant themes from the provided list. Only select themes that are clearly present or central to the dream.

Available Themes:
{', '.join(themes_list)}

Return your response as a JSON object containing a single key "tagged_dreams". The value should be a list of objects, where each object has a "dream_id" (string, matching the ID provided) and a "themes" (list of strings from the Available Themes list) key corresponding to each dream processed. Ensure every dream ID from the input batch is present in the output list, even if its themes list is empty.

Dream Batch:
{batch_dream_text}
"""
        try:
            response = ai(prompt, output_schema=TaggedDreamBatch)
            validated_batch_instance = None
            processed_batch_themes = [] # List to hold TaggedDream instances from this batch

            # 1. Check if it's already the correct Pydantic type
            if isinstance(response, TaggedDreamBatch):
                validated_batch_instance = response
            # 2. Check if it's a dict and try to convert
            elif isinstance(response, dict):
                logging.debug(f"Batch {i}: TaggedDreamBatch response was dict, attempting Pydantic conversion.")
                try:
                    validated_batch_instance = TaggedDreamBatch(**response)
                    logging.debug(f"Batch {i}: Successfully converted dict to TaggedDreamBatch.")
                except Exception as e: # Catch Pydantic ValidationError etc.
                    logging.warning(f"Batch {i}: Failed to convert dict to TaggedDreamBatch: {e}. Dict keys: {list(response.keys())}")
            # 3. Handle if neither or conversion failed
            else:
                 logging.warning(f"Batch {i}: Unexpected GPT response type or structure for TaggedDreamBatch: {type(response)}. Cannot process.")

            # Proceed if we have a valid instance and it contains tagged_dreams (and tagged_dreams is a list)
            if validated_batch_instance and isinstance(validated_batch_instance.tagged_dreams, list):
                 processed_batch_themes = validated_batch_instance.tagged_dreams # This is now a list of TaggedDream objects
            elif validated_batch_instance: # Valid instance but tagged_dreams missing/wrong type
                 logging.warning(f"Batch {i}: Validated TaggedDreamBatch instance had invalid 'tagged_dreams' attribute: {validated_batch_instance.tagged_dreams}")
                 # processed_batch_themes remains empty list

            # Store themes from the processed batch (now iterating over TaggedDream objects)
            for tagged_dream in processed_batch_themes:
                # Add extra check: ensure item in list is actually a TaggedDream instance
                # (in case the LLM put something weird in the list even if the outer structure was okay)
                if isinstance(tagged_dream, TaggedDream) and isinstance(tagged_dream.themes, list):
                     # Ensure dream_id exists in the original batch to prevent hallucinated IDs
                     if tagged_dream.dream_id in batch_df['id'].values:
                         all_tagged_themes[tagged_dream.dream_id] = tagged_dream.themes
                     else:
                         logging.warning(f"Batch {i}: GPT response included an unknown dream_id '{tagged_dream.dream_id}'. Ignoring.")
                else:
                    # This handles cases where an item in the tagged_dreams list wasn't a valid TaggedDream object
                    logging.warning(f"Batch {i}: Malformed tagged dream object found within the 'tagged_dreams' list: {tagged_dream}. Skipping.")

        except Exception as e:
            logging.warning(f"Batch {i}: Error processing batch with GPT (Exception: {e}). Skipping batch.")
            # Dreams in this batch won't be tagged if an error occurs here

    logging.info(f"Finished tagging batches. Collected themes for {len(all_tagged_themes)} out of {total_dreams_to_tag} dreams.")

    # Map the collected themes back to the DataFrame
    # Use .get with a default empty list to handle dreams that failed tagging
    df_to_tag['gpt_themes'] = df_to_tag['id'].apply(lambda dream_id: all_tagged_themes.get(dream_id, []))

    # Analyze theme prevalence
    logging.info("Analyzing theme prevalence between top and bottom deciles...")
    theme_prevalence = []

    # Ensure target column exists
    if 'is_synesthesia' not in df_to_tag.columns:
        logging.error("Target column 'is_synesthesia' not found in tagged dataframe.")
        return

    for theme in themes_list:
        # Calculate counts for contingency table
        top_has_theme = df_to_tag[(df_to_tag['score_decile'] == df_to_tag['score_decile'].max()) & (df_to_tag['gpt_themes'].apply(lambda x: theme in x))].shape[0]
        top_no_theme = df_to_tag[(df_to_tag['score_decile'] == df_to_tag['score_decile'].max()) & (~df_to_tag['gpt_themes'].apply(lambda x: theme in x))].shape[0]
        bottom_has_theme = df_to_tag[(df_to_tag['score_decile'] == 0) & (df_to_tag['gpt_themes'].apply(lambda x: theme in x))].shape[0]
        bottom_no_theme = df_to_tag[(df_to_tag['score_decile'] == 0) & (~df_to_tag['gpt_themes'].apply(lambda x: theme in x))].shape[0]

        total_top = top_has_theme + top_no_theme
        total_bottom = bottom_has_theme + bottom_no_theme

        if total_top == 0 or total_bottom == 0:
            logging.warning(f"Skipping theme '{theme}' due to zero counts in top or bottom decile.")
            continue

        prevalence_top = top_has_theme / total_top
        prevalence_bottom = bottom_has_theme / total_bottom

        # Chi-squared test
        contingency_table = [[top_has_theme, top_no_theme], [bottom_has_theme, bottom_no_theme]]
        try:
            chi2, p, _, _ = chi2_contingency(contingency_table)
            p_value = f"{p:.4f}"
            chi2_value = f"{chi2:.2f}"
        except ValueError as e: # Handle cases with low expected frequencies
            logging.warning(f"Could not perform Chi2 test for theme '{theme}': {e}")
            p_value = "N/A"
            chi2_value = "N/A"


        theme_prevalence.append({
            'Theme': theme,
            'Prevalence Top Decile (Synesthesia)': prevalence_top,
            'Prevalence Bottom Decile (Baseline)': prevalence_bottom,
            'Chi2': chi2_value,
            'p': p_value
        })

    prevalence_df = pd.DataFrame(theme_prevalence).sort_values('p')
    logging.info("GPT Theme Prevalence Comparison (Top vs Bottom Deciles):\n" + prevalence_df.to_string())

    # Qualitative Summary using LLM
    try:
        summary_prompt = f"""Based on the following theme prevalence data comparing dream reports from two groups (Top Decile vs Bottom Decile), please provide a brief qualitative summary highlighting the most significant thematic differences. Focus on themes with low p-values (e.g., < 0.05 or < 0.1).

Data:
{prevalence_df.to_string()}

Summary:"""
        summary_response = ai(summary_prompt)
        logging.info(f"LLM Qualitative Summary of Theme Differences:\n{summary_response}")
    except Exception as e:
        logging.error(f"Error generating LLM summary for themes: {e}")

    # Save tagged data (optional)
    # df_to_tag.to_csv("synesthesia_dreams_per_dream_tagged.csv", index=False)
    # logging.info("Saved GPT per-dream tagged data to synesthesia_dreams_per_dream_tagged.csv")


# --- GPT Batch Theme Analysis ---

class CommonThemes(BaseModel):
    """Structured output for common themes identified in a batch of dreams."""
    themes: List[str] = Field(description="List of 3-5 common themes found in the batch of dreams.")

# Pydantic models for theme grouping
class ThemeGroup(BaseModel):
    """Represents a group of related themes."""
    group_name: str = Field(description="A concise name for the theme group (e.g., 'Negative Emotions', 'Sensory Details').")
    original_themes: List[str] = Field(description="The list of original themes belonging to this group.")

class GroupedThemesOutput(BaseModel):
    """Structured output for the theme grouping task."""
    theme_groups: List[ThemeGroup] = Field(description="A list of theme groups, where each group contains related original themes.")

# Pydantic models for second-pass theme assignment
class ThemeAssignment(BaseModel):
    """Assigns an original theme to an existing group."""
    original_theme: str = Field(description="The theme that was previously unassigned.")
    assigned_group: str = Field(description="The name of the existing group it best fits into.")

class AssignThemesToGroupsOutput(BaseModel):
    """Structured output for assigning leftover themes to existing groups."""
    assignments: List[ThemeAssignment] = Field(description="List of assignments of original themes to existing group names.")

def analyze_gpt_batch_themes(df, api_key, batch_size=30, iterations=10):
    """ Analyzes batches of dreams from top/bottom deciles using GPT to find common themes. """
    logging.info(f"Running GPT Batch Theme analysis on top/bottom deciles ({iterations} iterations, batch size {batch_size})...")
    if not api_key:
        logging.error("OpenAI API key not provided. Skipping GPT Batch Theme analysis.")
        return
    if 'score_decile' not in df.columns or 'text' not in df.columns:
        logging.error("DataFrame must have 'score_decile' and 'text' columns.")
        return

    try:
        from simpleaichat import AIChat
    except ImportError:
        logging.error("simpleaichat library not installed. Skipping GPT Batch Theme analysis.")
        return

    # Filter for top and bottom deciles
    bottom_decile_df = df[df['score_decile'] == 0].copy()
    top_decile_df = df[df['score_decile'] == df['score_decile'].max()].copy()

    if bottom_decile_df.empty or top_decile_df.empty:
        logging.warning("Not enough data in top or bottom deciles for GPT batch theme analysis.")
        return

    ai = AIChat(console=False, model='gpt-4o', api_key=api_key)
    all_top_themes = collections.Counter()
    all_bottom_themes = collections.Counter()
    iteration_results = [] # To store results from each iteration for saving

    logging.info(f"Starting {iterations} iterations of batch theme analysis...")
    for i in range(iterations):
        logging.info(f"Iteration {i+1}/{iterations}")

        # Sample batches
        top_batch = top_decile_df.sample(n=min(batch_size, len(top_decile_df)), random_state=RANDOM_STATE + i)
        bottom_batch = bottom_decile_df.sample(n=min(batch_size, len(bottom_decile_df)), random_state=RANDOM_STATE + i)

        # Concatenate text (simple join, consider truncation if needed for context limits)
        # Using a max length to avoid overly long prompts
        max_prompt_chars = 15000
        top_text_batch = "\n---\n".join(top_batch['text'].astype(str))[:max_prompt_chars]
        bottom_text_batch = "\n---\n".join(bottom_batch['text'].astype(str))[:max_prompt_chars]

        # --- Process Top Decile Batch ---
        prompt_top = f"""Analyze the following batch of dream reports, separated by '---'. Identify and list 3-5 common or recurring themes present in this batch. Be concise.

Dream Batch:
{top_text_batch}"""
        normalized_themes_top = [] # Initialize for the iteration
        try:
            response_top = ai(prompt_top, output_schema=CommonThemes)
            validated_instance_top = None
            current_themes = []
            normalized_themes_top = [] # Initialize for the iteration

            # 1. Check if it's already the correct Pydantic type
            if isinstance(response_top, CommonThemes):
                validated_instance_top = response_top
            # 2. Check if it's a dict and try to convert
            elif isinstance(response_top, dict):
                logging.debug(f"Iteration {i+1}: Top themes response was dict, attempting Pydantic conversion.")
                try:
                    validated_instance_top = CommonThemes(**response_top)
                    logging.debug(f"Iteration {i+1}: Successfully converted dict to CommonThemes for top batch.")
                except Exception as e: # Catch Pydantic ValidationError etc.
                    logging.warning(f"Iteration {i+1}: Failed to convert dict to CommonThemes for top batch: {e}. Dict keys: {list(response_top.keys())}")
            # 3. Handle if neither or conversion failed
            else:
                 logging.warning(f"Iteration {i+1}: Unexpected GPT response type or structure for top batch: {type(response_top)}. Cannot process.")

            # Proceed if we have a valid instance
            if validated_instance_top and isinstance(validated_instance_top.themes, list):
                current_themes = validated_instance_top.themes
                # Normalize themes (lowercase, strip) before counting
                normalized_themes_top = [theme.lower().strip() for theme in current_themes]
                all_top_themes.update(normalized_themes_top)
                logging.debug(f"Iteration {i+1} Top Themes: {normalized_themes_top}")
            else:
                # Log if validation failed or themes attribute was wrong type/missing
                if validated_instance_top: # It was a valid instance but themes were bad
                     logging.warning(f"Iteration {i+1}: Validated CommonThemes instance for top batch had invalid 'themes' attribute: {validated_instance_top.themes}")
                # normalized_themes_top remains empty list if instance validation failed
                normalized_themes_top = []

        except Exception as e:
            logging.warning(f"Iteration {i+1}: Error processing top decile batch (Exception: {e}).")
            normalized_themes_top = [] # Ensure it's empty on outer error too

        # --- Process Bottom Decile Batch ---
        prompt_bottom = f"""Analyze the following batch of dream reports, separated by '---'. Identify and list 3-5 common or recurring themes present in this batch. Be concise.

Dream Batch:
{bottom_text_batch}"""
        normalized_themes_bottom = [] # Initialize for the iteration
        try:
            response_bottom = ai(prompt_bottom, output_schema=CommonThemes)
            validated_instance_bottom = None
            current_themes = []
            normalized_themes_bottom = [] # Initialize for the iteration

            # 1. Check if it's already the correct Pydantic type
            if isinstance(response_bottom, CommonThemes):
                validated_instance_bottom = response_bottom
            # 2. Check if it's a dict and try to convert
            elif isinstance(response_bottom, dict):
                logging.debug(f"Iteration {i+1}: Bottom themes response was dict, attempting Pydantic conversion.")
                try:
                    validated_instance_bottom = CommonThemes(**response_bottom)
                    logging.debug(f"Iteration {i+1}: Successfully converted dict to CommonThemes for bottom batch.")
                except Exception as e: # Catch Pydantic ValidationError etc.
                    logging.warning(f"Iteration {i+1}: Failed to convert dict to CommonThemes for bottom batch: {e}. Dict keys: {list(response_bottom.keys())}")
            # 3. Handle if neither or conversion failed
            else:
                 logging.warning(f"Iteration {i+1}: Unexpected GPT response type or structure for bottom batch: {type(response_bottom)}. Cannot process.")

            # Proceed if we have a valid instance
            if validated_instance_bottom and isinstance(validated_instance_bottom.themes, list):
                current_themes = validated_instance_bottom.themes
                # Normalize themes (lowercase, strip) before counting
                normalized_themes_bottom = [theme.lower().strip() for theme in current_themes]
                all_bottom_themes.update(normalized_themes_bottom)
                logging.debug(f"Iteration {i+1} Bottom Themes: {normalized_themes_bottom}")
            else:
                # Log if validation failed or themes attribute was wrong type/missing
                if validated_instance_bottom: # It was a valid instance but themes were bad
                     logging.warning(f"Iteration {i+1}: Validated CommonThemes instance for bottom batch had invalid 'themes' attribute: {validated_instance_bottom.themes}")
                # normalized_themes_bottom remains empty list if instance validation failed
                normalized_themes_bottom = []

        except Exception as e:
            logging.warning(f"Iteration {i+1}: Error processing bottom decile batch (Exception: {e}).")
            normalized_themes_bottom = [] # Ensure it's empty on outer error too

        # Append results for this iteration
        iteration_results.append({
            'iteration': i + 1,
            'top_themes': str(normalized_themes_top), # Store list as string
            'bottom_themes': str(normalized_themes_bottom) # Store list as string
        })

    logging.info("Finished batch theme analysis iterations.")

    # Save iteration results to CSV before aggregation
    iterations_df = pd.DataFrame(iteration_results)
    iterations_csv_path = 'synesthesia_gpt_batch_theme_iterations.csv'
    try:
        iterations_df.to_csv(iterations_csv_path, index=False)
        logging.info(f"Saved GPT batch theme iteration results to {iterations_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save batch theme iteration results: {e}")


    # Aggregate and display results
    top_themes_df = pd.DataFrame(all_top_themes.items(), columns=['Theme', 'Top Decile Count']).sort_values('Top Decile Count', ascending=False)
    bottom_themes_df = pd.DataFrame(all_bottom_themes.items(), columns=['Theme', 'Bottom Decile Count']).sort_values('Bottom Decile Count', ascending=False)

    # Merge results for comparison
    comparison_df = pd.merge(top_themes_df, bottom_themes_df, on='Theme', how='outer').fillna(0)
    comparison_df['Total Count'] = comparison_df['Top Decile Count'] + comparison_df['Bottom Decile Count']
    comparison_df = comparison_df.sort_values('Total Count', ascending=False)

    logging.info(f"Aggregated Raw Common Theme Counts ({iterations} iterations, batch size {batch_size}):\n" + comparison_df.to_string())

    # --- LLM Grouping Step ---
    logging.info("Attempting to group raw themes using LLM...")
    unique_themes = comparison_df['Theme'].unique().tolist()

    if not unique_themes:
        logging.warning("No unique themes found to group. Skipping grouping step.")
    else:
        grouping_prompt = f"""Given the following list of themes extracted from dream reports, please group them into meaningful semantic categories. Aim for a reasonable number of broad categories (e.g., 5-10). Ensure every original theme is assigned to exactly one group.

Themes to group:
{', '.join(unique_themes)}

Return the result as a JSON object structured according to the provided schema.
"""
        theme_to_group_map = {}
        grouped_comparison_df = pd.DataFrame() # Initialize empty df

        try:
            grouping_response = ai(grouping_prompt, output_schema=GroupedThemesOutput)
            validated_grouping_instance = None

            # 1. Check if it's already the correct Pydantic type
            if isinstance(grouping_response, GroupedThemesOutput):
                validated_grouping_instance = grouping_response
            # 2. Check if it's a dict and try to convert
            elif isinstance(grouping_response, dict):
                logging.debug("Grouping response was dict, attempting Pydantic conversion.")
                try:
                    validated_grouping_instance = GroupedThemesOutput(**grouping_response)
                    logging.debug("Successfully converted dict to GroupedThemesOutput.")
                except Exception as e: # Catch Pydantic ValidationError etc.
                    logging.warning(f"Failed to convert dict to GroupedThemesOutput: {e}. Dict keys: {list(grouping_response.keys())}")
            # 3. Handle if neither or conversion failed
            else:
                 logging.warning(f"Unexpected GPT response type or structure for grouping: {type(grouping_response)}. Cannot process.")

            # Proceed if we have a valid instance and it contains theme groups
            if validated_grouping_instance and isinstance(validated_grouping_instance.theme_groups, list): # Check inner list type too
                logging.info("Successfully received and validated theme grouping from LLM.")
                # Create a reverse map: {original_theme: group_name}
                # Use validated_grouping_instance here
                for group in validated_grouping_instance.theme_groups:
                    # Basic validation for the inner structure
                    if not isinstance(group, ThemeGroup) or not isinstance(group.original_themes, list):
                         logging.warning(f"Skipping malformed group object in grouping response: {group}")
                         continue
                    for original_theme in group.original_themes:
                        # Handle potential duplicates or overlaps from LLM - first assignment wins
                        if original_theme not in theme_to_group_map:
                            theme_to_group_map[original_theme] = group.group_name
                        else:
                            logging.warning(f"Theme '{original_theme}' assigned to multiple groups ('{theme_to_group_map[original_theme]}' and '{group.group_name}'). Keeping first assignment.")

                # Check if all original themes were mapped
                mapped_themes = set(theme_to_group_map.keys())
                unmapped_themes = [t for t in unique_themes if t not in mapped_themes]

                # --- Second Pass Assignment for Unmapped Themes ---
                if unmapped_themes and theme_to_group_map: # Only run if there are unmapped themes AND existing groups
                    existing_group_names = sorted(list(set(theme_to_group_map.values())))
                    logging.info(f"Attempting second pass assignment for {len(unmapped_themes)} unmapped themes into existing groups: {existing_group_names}")

                    assignment_prompt = f"""The following themes were not assigned to a group in the previous step. Please assign each theme to the *most appropriate* existing group from the list provided.

Themes to Assign:
{', '.join(unmapped_themes)}

Existing Group Names:
{', '.join(existing_group_names)}

Return the result as a JSON object structured according to the provided schema. Ensure every theme from 'Themes to Assign' is included in the output.
"""
                    try:
                        assignment_response = ai(assignment_prompt, output_schema=AssignThemesToGroupsOutput)
                        validated_assignment_instance = None

                        # Validate response (similar pattern to other calls)
                        if isinstance(assignment_response, AssignThemesToGroupsOutput):
                            validated_assignment_instance = assignment_response
                        elif isinstance(assignment_response, dict):
                            logging.debug("Second pass assignment response was dict, attempting Pydantic conversion.")
                            try:
                                validated_assignment_instance = AssignThemesToGroupsOutput(**assignment_response)
                                logging.debug("Successfully converted dict to AssignThemesToGroupsOutput.")
                            except Exception as e:
                                logging.warning(f"Failed to convert dict to AssignThemesToGroupsOutput: {e}. Dict keys: {list(assignment_response.keys())}")
                        else:
                            logging.warning(f"Unexpected response type for second pass assignment: {type(assignment_response)}")

                        # Process valid assignments
                        if validated_assignment_instance and isinstance(validated_assignment_instance.assignments, list):
                            assigned_in_pass_2 = 0
                            still_unmapped_after_pass_2 = list(unmapped_themes) # Start with all unmapped

                            for assignment in validated_assignment_instance.assignments:
                                if isinstance(assignment, ThemeAssignment) and assignment.original_theme in unmapped_themes:
                                    # Check if the assigned group actually exists
                                    if assignment.assigned_group in existing_group_names:
                                        theme_to_group_map[assignment.original_theme] = assignment.assigned_group
                                        logging.debug(f"Assigned '{assignment.original_theme}' to group '{assignment.assigned_group}' in second pass.")
                                        if assignment.original_theme in still_unmapped_after_pass_2:
                                            still_unmapped_after_pass_2.remove(assignment.original_theme)
                                        assigned_in_pass_2 += 1
                                    else:
                                        logging.warning(f"LLM assigned theme '{assignment.original_theme}' to a non-existent group '{assignment.assigned_group}' in second pass. Ignoring assignment.")
                                else:
                                     logging.warning(f"Malformed assignment object in second pass response: {assignment}")

                            logging.info(f"Successfully assigned {assigned_in_pass_2} themes in the second pass.")
                            if still_unmapped_after_pass_2:
                                logging.warning(f"Themes still unassigned after second pass: {still_unmapped_after_pass_2}. They will be excluded.")
                            unmapped_themes = still_unmapped_after_pass_2 # Update the list of truly unmapped themes
                        else:
                            logging.warning("LLM did not return valid assignments in the second pass. Unmapped themes remain excluded.")

                    except Exception as e:
                        logging.error(f"Error during second pass LLM assignment call: {e}. Unmapped themes remain excluded.")

                elif unmapped_themes: # Log if unmapped themes exist but no groups were formed initially
                     logging.warning(f"LLM did not assign groups for the following themes: {unmapped_themes}. No existing groups to assign them to. They will be excluded.")


                # Calculate grouped counts (using potentially updated theme_to_group_map)
                grouped_counts = collections.defaultdict(lambda: {'Top Decile Count': 0, 'Bottom Decile Count': 0})
                for _, row in comparison_df.iterrows():
                    original_theme = row['Theme']
                    if original_theme in theme_to_group_map:
                        group_name = theme_to_group_map[original_theme]
                        grouped_counts[group_name]['Top Decile Count'] += row['Top Decile Count']
                        grouped_counts[group_name]['Bottom Decile Count'] += row['Bottom Decile Count']

                # Convert to DataFrame
                grouped_list = []
                for group_name, counts in grouped_counts.items():
                    grouped_list.append({
                        'Theme Group': group_name,
                        'Top Decile Count': counts['Top Decile Count'],
                        'Bottom Decile Count': counts['Bottom Decile Count']
                    })
                grouped_comparison_df = pd.DataFrame(grouped_list)

                # Calculate Total Count immediately after creating the DataFrame
                if not grouped_comparison_df.empty:
                    grouped_comparison_df['Total Count'] = grouped_comparison_df['Top Decile Count'] + grouped_comparison_df['Bottom Decile Count']
                else:
                    # Ensure columns exist even if empty for consistency downstream, though stats won't run
                    grouped_comparison_df['Total Count'] = []


                # Handle case where grouped_list might be empty before stats
                if not grouped_comparison_df.empty:
                    # --- Add Statistical Analysis to Grouped Themes ---
                    total_top_themes = grouped_comparison_df['Top Decile Count'].sum()
                    total_bottom_themes = grouped_comparison_df['Bottom Decile Count'].sum()

                    # Check again for safety, although covered by outer check
                    if total_top_themes > 0 and total_bottom_themes > 0:
                        logging.info("Calculating proportions and running statistical tests for grouped themes...")
                        grouped_comparison_df['Prop Top'] = grouped_comparison_df['Top Decile Count'] / total_top_themes
                        grouped_comparison_df['Prop Bottom'] = grouped_comparison_df['Bottom Decile Count'] / total_bottom_themes

                        # Calculate ratio, handle division by zero
                        grouped_comparison_df['Ratio (Top/Bottom)'] = grouped_comparison_df.apply(
                            lambda row: row['Prop Top'] / row['Prop Bottom'] if row['Prop Bottom'] > 0 else np.inf, axis=1
                        )

                        # Perform Chi-squared test for each group
                        p_values = []
                        for index, row in grouped_comparison_df.iterrows():
                            group_top_count = row['Top Decile Count']
                            group_bottom_count = row['Bottom Decile Count']
                            other_top_count = total_top_themes - group_top_count
                            other_bottom_count = total_bottom_themes - group_bottom_count

                            # Create 2x2 contingency table: [[GroupTop, OtherTop], [GroupBottom, OtherBottom]]
                            contingency_table = [[group_top_count, other_top_count],
                                                 [group_bottom_count, other_bottom_count]]
                            try:
                                chi2, p, _, _ = chi2_contingency(contingency_table)
                                p_values.append(f"{p:.4f}")
                            except ValueError as e: # Handle cases with low expected frequencies or zero counts
                                logging.warning(f"Could not perform Chi2 test for theme group '{row['Theme Group']}': {e}")
                                p_values.append("N/A")

                        grouped_comparison_df['p-value'] = p_values
                        # Sort by p-value primarily, then by total count
                        grouped_comparison_df = grouped_comparison_df.sort_values(by=['p-value', 'Total Count'], ascending=[True, False])

                    else:
                        logging.warning("Cannot calculate proportions or run tests due to zero total themes in top or bottom decile groups.")
                        # Add empty columns if tests couldn't run
                        grouped_comparison_df['Prop Top'] = np.nan
                        grouped_comparison_df['Prop Bottom'] = np.nan
                        grouped_comparison_df['Ratio (Top/Bottom)'] = np.nan
                        grouped_comparison_df['p-value'] = "N/A"
                        # Fallback sort (Total Count already exists)
                        grouped_comparison_df = grouped_comparison_df.sort_values('Total Count', ascending=False)


                    logging.info("Grouped Theme Counts with Stats:\n" + grouped_comparison_df.to_string())
                else:
                    logging.warning("No valid groups were formed after processing LLM response.")
                    # grouped_comparison_df remains empty

            else:
                # This covers: invalid type, conversion failure, or valid instance but empty/missing/invalid theme_groups
                logging.warning("LLM did not return valid or parsable theme groupings. Skipping grouped analysis.")
                # Ensure grouped_comparison_df remains empty if grouping fails
                grouped_comparison_df = pd.DataFrame()


        except Exception as e:
            logging.error(f"Error during LLM theme grouping call or processing: {e}. Skipping grouped analysis.")
            grouped_comparison_df = pd.DataFrame() # Ensure it's empty on outer error


    # --- Original Qualitative Summary (Based on Raw Themes) ---
    try:
        summary_prompt_raw = f"""Based on the following aggregated RAW theme counts from analyzing batches of dream reports from two groups (Top Decile vs Bottom Decile), please provide a brief qualitative summary highlighting the most prominent themes for each group and any notable differences.

Data (Theme, Top Decile Count, Bottom Decile Count):
{comparison_df.to_string()}

Summary:"""
        summary_response_raw = ai(summary_prompt_raw)
        logging.info(f"LLM Qualitative Summary of RAW Batch Theme Differences:\n{summary_response_raw}")
    except Exception as e:
        logging.error(f"Error generating LLM summary for RAW batch themes: {e}")


    # --- New Qualitative Summary (Based on Grouped Themes, if available) ---
    if not grouped_comparison_df.empty and 'p-value' in grouped_comparison_df.columns: # Check if stats were added
        try:
            # Prepare data string for prompt, ensuring relevant columns exist
            stats_columns = ['Theme Group', 'Top Decile Count', 'Bottom Decile Count', 'Prop Top', 'Prop Bottom', 'Ratio (Top/Bottom)', 'p-value']
            available_cols = [col for col in stats_columns if col in grouped_comparison_df.columns]
            data_string_for_prompt = grouped_comparison_df[available_cols].to_string()

            summary_prompt_grouped = f"""Based on the following GROUPED theme counts and statistical analysis from comparing dream reports between two groups (Top Decile vs Bottom Decile), please provide a brief qualitative summary.

Highlight the most prominent theme groups overall and for each group. Most importantly, focus on the theme groups showing statistically significant differences (e.g., low p-value < 0.05 or 0.1) and describe the nature of these differences (e.g., "Group X was significantly more prevalent in the Top Decile"). Use the proportions and ratios to quantify the differences where appropriate.

Data ({', '.join(available_cols)}):
{data_string_for_prompt}

Summary:"""
            summary_response_grouped = ai(summary_prompt_grouped)
            logging.info(f"LLM Qualitative Summary of GROUPED Batch Theme Differences (with Stats):\n{summary_response_grouped}")
        except Exception as e:
            logging.error(f"Error generating LLM summary for GROUPED batch themes with stats: {e}")
    elif not grouped_comparison_df.empty: # Fallback if stats weren't calculated but groups exist
         try:
            logging.warning("Generating grouped theme summary without statistical details as they were not calculated.")
            summary_prompt_grouped_no_stats = f"""Based on the following GROUPED theme counts from analyzing batches of dream reports from two groups (Top Decile vs Bottom Decile), please provide a brief qualitative summary highlighting the most prominent theme groups for each group and any notable differences. Statistical analysis was not available.

Data (Theme Group, Top Decile Count, Bottom Decile Count):
{grouped_comparison_df[['Theme Group', 'Top Decile Count', 'Bottom Decile Count']].to_string()}

Summary:"""
            summary_response_grouped = ai(summary_prompt_grouped_no_stats)
            logging.info(f"LLM Qualitative Summary of GROUPED Batch Theme Differences (No Stats):\n{summary_response_grouped}")
         except Exception as e:
            logging.error(f"Error generating LLM summary for GROUPED batch themes (no stats): {e}")


    # Return both raw and grouped results (if available)
    return comparison_df, grouped_comparison_df


# --- Keyword Theme Analysis ---

def analyze_keyword_themes(df, keyword_csv_path='synesthesia/syn_themes_V2.csv', api_key=None):
    """
    Analyzes dream text based on predefined keyword lists from a CSV,
    comparing prevalence between actual synesthesia (True) and baseline (False) groups.
    """
    logging.info(f"Running Keyword Theme analysis using definitions from: {keyword_csv_path}")
    # Check for required columns: 'is_synesthesia' and 'text'
    if 'is_synesthesia' not in df.columns or 'text' not in df.columns:
        logging.error("DataFrame must have 'is_synesthesia' and 'text' columns for keyword theme analysis.")
        return
    if not os.path.exists(keyword_csv_path):
        logging.error(f"Keyword theme definition file not found: {keyword_csv_path}")
        return

    # 1. Load Keyword Definitions
    try:
        theme_defs_df = pd.read_csv(keyword_csv_path)
        if not all(col in theme_defs_df.columns for col in ['Theme', 'Associated Words']):
            logging.error(f"Keyword CSV must contain 'Theme' and 'Associated Words' columns.")
            return

        # Parse keywords into sets of lowercase words
        theme_keywords = {}
        for _, row in theme_defs_df.iterrows():
            theme_name = row['Theme']
            words_str = row['Associated Words']
            if pd.isna(words_str):
                logging.warning(f"No associated words found for theme '{theme_name}'. Skipping.")
                continue
            # Split, strip whitespace, convert to lowercase, handle potential empty strings
            keywords = {word.strip().lower() for word in words_str.split(',') if word.strip()}
            if keywords:
                theme_keywords[theme_name] = keywords
            else:
                 logging.warning(f"No valid keywords parsed for theme '{theme_name}' from string '{words_str}'. Skipping.")

        if not theme_keywords:
            logging.error("No valid theme definitions loaded from CSV.")
            return
        logging.info(f"Loaded {len(theme_keywords)} keyword theme definitions.")

    except Exception as e:
        logging.error(f"Error loading or parsing keyword theme CSV: {e}")
        return

    # 2. Preprocess Text (Tokenize and Lowercase)
    # Reuse NLTK tokenization, no need for stopword removal here
    logging.info("Preprocessing text for keyword matching...")
    try:
        # Create sets of lowercase tokens for faster lookups
        df['processed_tokens'] = df['text'].fillna('').apply(
            lambda x: {word.lower() for word in word_tokenize(str(x)) if word.isalpha()}
        )
    except Exception as e:
        logging.error(f"Error during text preprocessing for keyword analysis: {e}")
        return

    # 3. Tag Dreams based on Keywords
    logging.info("Tagging dreams based on keyword presence...")
    df['keyword_themes'] = df['processed_tokens'].apply(
        lambda tokens: [theme for theme, keywords in theme_keywords.items() if not keywords.isdisjoint(tokens)]
    )

    # 4. Filter by Ground Truth Label ('is_synesthesia')
    syn_true_df = df[df['is_synesthesia'] == True].copy()
    syn_false_df = df[df['is_synesthesia'] == False].copy()

    if syn_true_df.empty or syn_false_df.empty:
        logging.warning("Not enough data in is_synesthesia=True or is_synesthesia=False groups for keyword theme analysis.")
        return

    # 5. Analyze Theme Prevalence
    logging.info("Analyzing keyword theme prevalence between is_synesthesia=True and is_synesthesia=False groups...")
    theme_prevalence = []
    total_syn_true = len(syn_true_df)
    total_syn_false = len(syn_false_df)

    for theme in theme_keywords.keys():
        # Calculate counts for contingency table based on ground truth
        true_has_theme = syn_true_df['keyword_themes'].apply(lambda x: theme in x).sum()
        false_has_theme = syn_false_df['keyword_themes'].apply(lambda x: theme in x).sum()

        true_no_theme = total_syn_true - true_has_theme
        false_no_theme = total_syn_false - false_has_theme

        # Calculate prevalence for each group
        prevalence_true = true_has_theme / total_syn_true if total_syn_true > 0 else 0
        prevalence_false = false_has_theme / total_syn_false if total_syn_false > 0 else 0

        # Chi-squared test using ground truth counts
        contingency_table = [[true_has_theme, true_no_theme], [false_has_theme, false_no_theme]]
        try:
            # Add correction=False if needed, especially for small counts, though default is usually fine.
            chi2, p, _, _ = chi2_contingency(contingency_table, correction=False)
            p_value = f"{p:.4f}"
            chi2_value = f"{chi2:.2f}"
        except ValueError as e: # Handle cases with low expected frequencies
            logging.warning(f"Could not perform Chi2 test for keyword theme '{theme}': {e}")
            p_value = "N/A"
            chi2_value = "N/A"

        theme_prevalence.append({
            'Theme': theme,
            'Prevalence Synesthesia (True)': prevalence_true,
            'Prevalence Baseline (False)': prevalence_false,
            'Chi2': chi2_value,
            'p': p_value
        })

    prevalence_df = pd.DataFrame(theme_prevalence).sort_values('p')
    logging.info("Keyword Theme Prevalence Comparison (is_synesthesia=True vs False):\n" + prevalence_df.to_string())

    # 6. Optional LLM Summary
    if api_key:
        try:
            ai = AIChat(console=False, model='gpt-4o', api_key=api_key)
            summary_prompt = f"""Based on the following keyword-based theme prevalence data comparing dream reports from two groups based on their actual label (is_synesthesia=True vs is_synesthesia=False), please provide a brief qualitative summary highlighting the most significant thematic differences. Focus on themes with low p-values (e.g., < 0.05 or < 0.1).

Data:
{prevalence_df.to_string()}

Summary:"""
            summary_response = ai(summary_prompt)
            logging.info(f"LLM Qualitative Summary of Keyword Theme Differences:\n{summary_response}")
        except ImportError:
            logging.warning("simpleaichat not installed. Skipping LLM summary for keyword themes.")
        except Exception as e:
            logging.error(f"Error generating LLM summary for keyword themes: {e}")

    # Clean up added columns from the original DataFrame passed into the function
    # Use errors='ignore' in case the function exited early before columns were added
    df.drop(columns=['processed_tokens', 'keyword_themes'], inplace=True, errors='ignore')


# --- BERTopic Theme Analysis ---

def analyze_bertopic_themes(df, model_path=BERTOPIC_MODEL_PATH, api_key=None, min_count_threshold: int = 5):
    """
    Analyzes dream text using a pre-trained BERTopic model via DreamTopicAnalyzer,
    comparing topic prevalence between actual synesthesia (True) and baseline (False) groups.
    Filters results to include topics appearing at least `min_count_threshold` times in either group.

    Returns:
        pd.DataFrame: DataFrame containing topic prevalence comparison results, or None if analysis fails.
    """
    logging.info(f"Running BERTopic Theme analysis using model from: {model_path}")
    # Check for required columns: 'is_synesthesia', 'text', and 'id'
    if not all(col in df.columns for col in ['is_synesthesia', 'text', 'id']):
        logging.error("DataFrame must have 'is_synesthesia', 'text', and 'id' columns for BERTopic analysis.")
        return None, None, None # Return tuple on failure

    try:
        # 1. Instantiate the Analyzer (loads the model)
        analyzer = DreamTopicAnalyzer(model_path=model_path)

        # 2. Prepare Sentences
        # Pass the label column to carry it through
        sentences_df = analyzer.prepare_sentences_from_dataframe(df, text_col='text', id_col='id', label_col='is_synesthesia')
        if sentences_df.empty:
             logging.warning("No sentences were generated. Skipping BERTopic analysis.")
             return None, None, None # Return tuple on failure

        # 3. Infer Topics for Sentences
        sentences_with_topics_df = analyzer.infer_topics_for_sentences(sentences_df)
        if sentences_with_topics_df.empty or 'topic_id' not in sentences_with_topics_df.columns:
             logging.warning("No topics could be inferred or only outlier topics found. Skipping BERTopic analysis.")
             # Pass back the analyzer even if topics failed. Grouping needs the model AND the df_merged with topics.
             # If topic inference fails, df_merged won't have topics, so grouping can't proceed anyway.
             # Return None for the dataframes.
             return None, None, analyzer # Return tuple on failure

        # 4. Aggregate Topics back to Dream Level
        # Pass the original df to merge back all columns
        df_merged = analyzer.aggregate_topics_to_dreams(sentences_with_topics_df, df, doc_id_col='doc_id', original_id_col='id', topic_col='topic_id')
        # df_merged now contains original columns + 'topic_id' (set of topics per dream)

    except FileNotFoundError:
        # Error already logged by DreamTopicAnalyzer._load_model
        return None, None, None # Return tuple on failure
    except (RuntimeError, ValueError, Exception) as e:
        logging.error(f"Error during BERTopic analysis steps: {e}")
        # Try returning the analyzer object even on error? Might be risky.
        # Let's stick to returning None for the dataframes.
        return None, None, None # Return tuple on failure


    # 5. Analyze Topic Prevalence by Group (is_synesthesia True vs False)
    # Use the df_merged which has the aggregated topics per dream
    logging.info("Analyzing BERTopic prevalence between is_synesthesia=True and is_synesthesia=False groups...")
    try:
        topic_info_df = analyzer.get_topic_info(filter_outliers=True) # Get topic info from the analyzer
        all_topics = topic_info_df['Topic'].unique() # Get all valid topic IDs from the model info
    except RuntimeError as e:
        logging.error(f"Could not get topic info: {e}")
        return None, None, analyzer # Return tuple on failure, keep analyzer

    syn_true_df = df_merged[df_merged['is_synesthesia'] == True]
    syn_false_df = df_merged[df_merged['is_synesthesia'] == False]
    total_syn_true = len(syn_true_df)
    total_syn_false = len(syn_false_df)

    if total_syn_true == 0 or total_syn_false == 0:
        logging.warning("Not enough data in is_synesthesia=True or is_synesthesia=False groups for BERTopic analysis.")
        return None, df_merged, analyzer # Return tuple on failure, keep analyzer and merged df

    topic_prevalence = []

    for topic_id in tqdm(all_topics, desc="Comparing topic prevalence"):
        # Count how many dreams in each group contain this topic_id
        true_has_topic = syn_true_df['topic_id'].apply(lambda topics: topic_id in topics).sum()
        false_has_topic = syn_false_df['topic_id'].apply(lambda topics: topic_id in topics).sum()

        true_no_topic = total_syn_true - true_has_topic
        false_no_topic = total_syn_false - false_has_topic

        # Calculate prevalence (proportion of dreams in the group containing the topic)
        prevalence_true = true_has_topic / total_syn_true
        prevalence_false = false_has_topic / total_syn_false

        # Chi-squared test
        contingency_table = [[true_has_topic, true_no_topic], [false_has_topic, false_no_topic]]
        try:
            chi2, p, _, _ = chi2_contingency(contingency_table, correction=False)
            p_value = f"{p:.4f}"
            chi2_value = f"{chi2:.2f}"
        except ValueError as e:
            # Log as debug because low expected frequencies are common and expected
            logging.debug(f"Could not perform Chi2 test for BERTopic ID {topic_id} (likely low counts): {e}")
            p_value = "N/A" # Still mark as N/A in the output
            chi2_value = "N/A"

        # Get topic representation (top words) from the topic_info_df
        topic_row = topic_info_df[topic_info_df['Topic'] == topic_id]
        if not topic_row.empty:
            # Representation might be list or string depending on BERTopic version/saving
            representation = topic_row['Representation'].iloc[0]
            if isinstance(representation, list):
                topic_words_str = ", ".join(representation)
            elif isinstance(representation, str):
                 topic_words_str = representation # Assume already formatted if string
            else:
                 topic_words_str = "N/A"
        else:
            topic_words_str = "N/A" # Should not happen if all_topics comes from topic_info_df


        topic_prevalence.append({
            'Topic ID': topic_id,
            'Top Words': topic_words_str,
            'Prevalence Synesthesia (True)': prevalence_true,
            'Prevalence Baseline (False)': prevalence_false,
            'Chi2': chi2_value,
            'p': p_value,
            'Count Syn (True)': true_has_topic, # Add counts for context
            'Count Base (False)': false_has_topic
        })

    prevalence_df = pd.DataFrame(topic_prevalence)

    # Filter based on minimum count threshold
    initial_topic_count = len(prevalence_df)
    prevalence_df = prevalence_df[
        (prevalence_df['Count Syn (True)'] >= min_count_threshold) |
        (prevalence_df['Count Base (False)'] >= min_count_threshold)
    ]
    filtered_topic_count = len(prevalence_df)
    logging.info(f"Filtered BERTopic results from {initial_topic_count} to {filtered_topic_count} topics based on min count threshold ({min_count_threshold}) in either group.")

    # Sort the filtered DataFrame by p-value
    prevalence_df = prevalence_df.sort_values('p')

    # Log only the head or significant results at INFO level
    logging.info(f"Filtered BERTopic Prevalence Comparison (Top 20 significant topics by p-value):\n{prevalence_df.head(20).to_string()}")
    # Log the full filtered dataframe at DEBUG level
    logging.debug(f"Full Filtered BERTopic Prevalence Comparison (min_count={min_count_threshold}):\n" + prevalence_df.to_string())


    # 6. Optional LLM Summary
    if api_key:
        logging.info("Preparing data for LLM qualitative summary...")
        try:
            # --- Prepare data specifically for LLM summary ---
            llm_df = prevalence_df.copy()

            # Convert p-value to numeric for filtering, coercing errors
            llm_df['p_numeric'] = pd.to_numeric(llm_df['p'], errors='coerce')

            # Calculate total count
            llm_df['Total Count'] = llm_df['Count Syn (True)'] + llm_df['Count Base (False)']

            # Apply LLM-specific filters
            llm_df_filtered = llm_df[
                (llm_df['Total Count'] >= 10) &
                (llm_df['p_numeric'] < 0.2)
            ].copy() # Use copy to avoid SettingWithCopyWarning

            logging.info(f"Found {len(llm_df_filtered)} topics meeting LLM criteria (Total Count >= 10, p < 0.2).")

            if not llm_df_filtered.empty:
                # Split into overrepresented groups
                syn_overrep = llm_df_filtered[llm_df_filtered['Prevalence Synesthesia (True)'] > llm_df_filtered['Prevalence Baseline (False)']]
                base_overrep = llm_df_filtered[llm_df_filtered['Prevalence Baseline (False)'] >= llm_df_filtered['Prevalence Synesthesia (True)']] # Include equality here

                # Select relevant columns for the prompt
                cols_to_show = ['Topic ID', 'Top Words', 'Prevalence Synesthesia (True)', 'Prevalence Baseline (False)', 'p']
                syn_overrep_str = syn_overrep[cols_to_show].to_string(index=False)
                base_overrep_str = base_overrep[cols_to_show].to_string(index=False)

                # --- Construct new LLM prompt ---
                summary_prompt = f"""Analyze the following BERTopic results comparing dreams from Synesthesia users (True) vs a Baseline group (False). The data shows topics filtered by statistical significance (p < 0.2) and minimum occurrence (Total Count >= 10).

The topics are split into two groups:
1. Topics MORE prevalent in the Synesthesia group.
2. Topics MORE prevalent in the Baseline group.

Based *only* on the provided data, provide a brief qualitative summary for EACH group separately. Describe the key themes that appear to differentiate the groups, using the 'Top Words' to interpret the topics.

--- Topics More Prevalent in Synesthesia Group (True) ---
{syn_overrep_str}

--- Topics More Prevalent in Baseline Group (False) ---
{base_overrep_str}

--- Summary ---
Synesthesia Group Themes: [Your summary focusing on themes from the first list]

Baseline Group Themes: [Your summary focusing on themes from the second list]
"""
                # --- Call LLM ---
                try:
                    ai = AIChat(console=False, model='gpt-4o', api_key=api_key) # Use a capable model like gpt-4o
                    summary_response = ai(summary_prompt)
                    logging.info(f"LLM Qualitative Summary of BERTopic Differences (Split by Group):\n{summary_response}")
                except ImportError:
                    logging.warning("simpleaichat not installed. Skipping LLM summary for BERTopic themes.")
                except Exception as e:
                    logging.error(f"Error calling LLM for BERTopic summary: {e}")
            else:
                logging.info("No topics met the criteria for LLM summary after filtering.")

        except Exception as e:
            logging.error(f"Error preparing data or prompting LLM for BERTopic summary: {e}")

    # Save the original results DataFrame (filtered only by min_count_threshold) to a CSV file
    try:
        bertopic_csv_path = 'synesthesia/synesthesia_bertopic_theme_prevalence.csv'
        prevalence_df.to_csv(bertopic_csv_path, index=False)
        logging.info(f"Saved BERTopic theme prevalence results to {bertopic_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save BERTopic results to CSV: {e}")

    # Return the prevalence df, the merged df with topics per dream, and the analyzer object
    return prevalence_df, df_merged, analyzer


# --- BERTopic Topic Grouping ---

def group_bertopic_topics(topic_model: BERTopic, embedding_model_name: str = 'all-mpnet-base-v2') -> Dict[int, int]:
    """
    Groups fine-grained BERTopic topics into broader themes using weighted embeddings, UMAP, and K-Means.

    Args:
        topic_model: The fitted BERTopic model object.
        embedding_model_name: The SentenceTransformer model to use for word embeddings ('all-mpnet-base-v2' recommended by paper).

    Returns:
        A dictionary mapping original BERTopic topic IDs to theme cluster IDs (0 to K-1).
        Returns an empty dictionary if grouping fails.
    """
    logging.info("Starting BERTopic topic grouping process...")
    if topic_model is None:
        logging.error("BERTopic model object is None. Cannot perform grouping.")
        return {}

    # --- Device Setup ---
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info(f"Using device for word embedding model: {device}")

    # --- Step 1: Initialize Sentence Transformer Model ---
    try:
        logging.info(f"Loading word embedding model: {embedding_model_name}")
        word_embedding_model = SentenceTransformer(embedding_model_name, device=device)
        embedding_dim = word_embedding_model.get_sentence_embedding_dimension()
        logging.info(f"Word embedding model loaded. Embedding dimension: {embedding_dim}")
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model '{embedding_model_name}': {e}")
        return {}

    # --- Step 2: Extract Topic Representations ---
    topics_info = {}
    try:
        all_topic_ids = sorted(topic_model.get_topic_freq()['Topic'].unique())
        valid_topic_ids = [tid for tid in all_topic_ids if tid != -1] # Exclude outlier topic
        num_topics = len(valid_topic_ids)
        logging.info(f"Found {num_topics} valid BERTopic topics (excluding outliers).")

        if num_topics == 0:
            logging.warning("No valid topics found in the BERTopic model. Cannot perform grouping.")
            return {}

        for topic_id in valid_topic_ids:
            # Get top 10 words and scores (c-TF-IDF or probabilities)
            topic_representation = topic_model.get_topic(topic_id)
            if topic_representation:
                top_words_scores = topic_representation[:10] # Ensure only top 10
                # Basic validation: check if scores are present and numeric. Accept list or tuple for inner items.
                if all(isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (int, float)) for item in top_words_scores):
                    topics_info[topic_id] = top_words_scores
                else:
                     logging.warning(f"Invalid representation format for topic {topic_id}. Skipping. Representation: {topic_representation}")
            else:
                logging.warning(f"Could not retrieve representation for topic {topic_id}. Skipping.")

        if not topics_info:
            logging.error("Could not extract valid representations for any topic. Aborting grouping.")
            return {}

    except Exception as e:
        logging.error(f"Error extracting topic representations from BERTopic model: {e}")
        return {}

    # --- Step 3: Calculate Weighted Topic Embeddings ---
    topic_embeddings = []
    topic_ids_in_order = [] # Keep track of the order for later mapping

    logging.info("Calculating weighted topic embeddings...")
    for topic_id in tqdm(valid_topic_ids, desc="Calculating topic embeddings"):
        if topic_id in topics_info:
            words = [item[0] for item in topics_info[topic_id]]
            weights = np.array([item[1] for item in topics_info[topic_id]])

            # Ensure weights are positive for np.average; c-TF-IDF can be negative sometimes, shift if needed.
            if np.any(weights < 0):
                logging.debug(f"Shifting weights for topic {topic_id} to be non-negative for averaging.")
                weights -= weights.min() # Shift minimum to 0

            # Handle case where all weights might become zero after shifting (unlikely but possible)
            if np.sum(weights) == 0:
                 logging.warning(f"All weights for topic {topic_id} are zero after shifting. Using uniform weights.")
                 weights = np.ones_like(weights) # Fallback to uniform weights


            try:
                word_embeddings = word_embedding_model.encode(words) # Shape: (num_words, embedding_dim)
                # Calculate weighted average
                topic_embedding = np.average(word_embeddings, axis=0, weights=weights)
                topic_embeddings.append(topic_embedding)
                topic_ids_in_order.append(topic_id)
            except Exception as e:
                logging.warning(f"Could not calculate embedding for topic {topic_id}. Skipping. Error: {e}")

    if not topic_embeddings:
        logging.error("Failed to calculate any topic embeddings. Aborting grouping.")
        return {}

    topic_embeddings_array = np.array(topic_embeddings) # Shape: (num_valid_topics, embedding_dim)
    logging.info(f"Calculated {len(topic_embeddings_array)} topic embeddings.")

    # --- Step 4: Standardize and Reduce Dimensionality with UMAP ---
    try:
        logging.info("Standardizing topic embeddings...")
        scaler = StandardScaler()
        standardized_embeddings = scaler.fit_transform(topic_embeddings_array)

        logging.info("Applying UMAP for dimensionality reduction...")
        umap_reducer = umap.UMAP(
            n_components=10,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=RANDOM_STATE
        )
        reduced_embeddings = umap_reducer.fit_transform(standardized_embeddings) # Shape: (num_valid_topics, 10)
        logging.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    except Exception as e:
        logging.error(f"Error during standardization or UMAP reduction: {e}")
        return {}

    # --- Step 5: Cluster Reduced Embeddings with K-Means ---
    try:
        logging.info("Clustering reduced embeddings using K-Means (K=20)...")
        num_themes = 20 # K=20 as specified in the paper
        kmeans_clusterer = KMeans(
            n_clusters=num_themes,
            random_state=RANDOM_STATE,
            n_init=10
        )
        theme_labels = kmeans_clusterer.fit_predict(reduced_embeddings) # Assigns theme label (0 to K-1)
        logging.info("K-Means clustering complete.")
    except Exception as e:
        logging.error(f"Error during K-Means clustering: {e}")
        return {}

    # --- Map original topic IDs to theme labels ---
    topic_to_theme_map = dict(zip(topic_ids_in_order, theme_labels))
    logging.info(f"Successfully mapped {len(topic_to_theme_map)} BERTopic topics to {num_themes} themes.")

    # --- Step 6: Generate Theme Representations ---
    logging.info("Generating representations for the 20 themes...")
    theme_details = collections.defaultdict(lambda: {'topic_ids': [], 'word_scores': collections.Counter()})
    for topic_id, theme_id in topic_to_theme_map.items():
        if topic_id in topics_info: # Ensure we have info for this topic
            theme_details[theme_id]['topic_ids'].append(topic_id)
            for word, score in topics_info[topic_id]:
                # Aggregate scores (simple sum for now)
                theme_details[theme_id]['word_scores'][word] += score

    theme_representations = {}
    num_theme_words = 10 # Number of top words to show per theme
    for theme_id, details in theme_details.items():
        # Get the top N words based on aggregated score
        top_words = [word for word, score in details['word_scores'].most_common(num_theme_words)]
        theme_representations[theme_id] = ", ".join(top_words)
        logging.debug(f"Theme {theme_id} (Topics: {details['topic_ids']}): Top words = {theme_representations[theme_id]}")

    logging.info(f"Generated representations for {len(theme_representations)} themes.")

    # Return both the mapping and the representations
    return topic_to_theme_map, theme_representations


# --- LLM Theme Labeling ---

class ThemeInput(BaseModel):
    """Input structure for a single theme to be labeled."""
    theme_id: int
    top_words: str

class ThemeLabelOutput(BaseModel):
    """Output structure for a single labeled theme."""
    theme_id: int
    theme_label: str = Field(description="A concise 1-2 word label for the theme.")

class LabeledThemesBatch(BaseModel):
    """Structured output for a batch of labeled themes."""
    labeled_themes: List[ThemeLabelOutput]


def get_llm_theme_labels(theme_representations: Dict[int, str], api_key: str) -> Dict[int, str]:
    """
    Uses an LLM to generate concise labels for themes based on their top words.

    Args:
        theme_representations: Dictionary mapping theme ID to a string of top words.
        api_key: OpenAI API key.

    Returns:
        Dictionary mapping theme ID to a concise LLM-generated label, or empty dict on failure.
    """
    logging.info("Generating concise theme labels using LLM...")
    if not theme_representations:
        logging.warning("No theme representations provided to generate labels.")
        return {}
    if not api_key:
        logging.error("OpenAI API key required for LLM theme labeling.")
        return {}

    try:
        ai = AIChat(console=False, model='gpt-4o', api_key=api_key)
    except ImportError:
        logging.error("simpleaichat not installed. Cannot generate LLM theme labels.")
        return {}
    except Exception as e:
        logging.error(f"Failed to initialize AIChat for theme labeling: {e}")
        return {}

    # Prepare input data for the prompt
    theme_input_list = []
    for theme_id, top_words in theme_representations.items():
        theme_input_list.append(ThemeInput(theme_id=theme_id, top_words=top_words))

    # Format the input for the prompt (simple list for now)
    prompt_input_str = "\n".join([f"Theme ID {t.theme_id}: {t.top_words}" for t in theme_input_list])

    prompt = f"""Many granular topics have been aggregated into only 20 grouped themes. Come up with a single concise 1-2 word label for each theme listed below.
The current "names" are just the top 10 most overrepresented words in this topic theme vs. expected. So come up with a 1 or 2-word name that is most likely to be the "essence" that this theme captures.

Themes to Label:
{prompt_input_str}

Return your response as a JSON object containing a single key "labeled_themes". The value should be a list of objects, where each object has a "theme_id" (integer, matching the input ID) and a "theme_label" (string, the 1-2 word label you generated) key corresponding to each theme processed. Ensure every theme ID from the input list is present in the output list.
"""

    theme_labels = {}
    try:
        response = ai(prompt, output_schema=LabeledThemesBatch)
        validated_instance = None

        # Validate response
        if isinstance(response, LabeledThemesBatch):
            validated_instance = response
        elif isinstance(response, dict):
            logging.debug("Theme labeling response was dict, attempting Pydantic conversion.")
            try:
                validated_instance = LabeledThemesBatch(**response)
                logging.debug("Successfully converted dict to LabeledThemesBatch.")
            except Exception as e:
                logging.warning(f"Failed to convert dict to LabeledThemesBatch: {e}. Dict keys: {list(response.keys())}")
        else:
            logging.warning(f"Unexpected GPT response type for theme labeling: {type(response)}. Cannot process.")

        # Process valid labels
        if validated_instance and isinstance(validated_instance.labeled_themes, list):
            for item in validated_instance.labeled_themes:
                if isinstance(item, ThemeLabelOutput) and isinstance(item.theme_label, str) and item.theme_label.strip():
                    # Check if the theme_id exists in the original input
                    if item.theme_id in theme_representations:
                        theme_labels[item.theme_id] = item.theme_label.strip()
                    else:
                         logging.warning(f"LLM returned label for unknown theme_id {item.theme_id}. Ignoring.")
                else:
                    logging.warning(f"Malformed labeled theme object found in LLM response: {item}. Skipping.")
            logging.info(f"Successfully generated labels for {len(theme_labels)} themes.")
        elif validated_instance:
            logging.warning(f"Validated LabeledThemesBatch instance had invalid 'labeled_themes' attribute: {validated_instance.labeled_themes}")
        # If validation failed entirely, theme_labels remains empty

    except Exception as e:
        logging.error(f"Error during LLM theme labeling call or processing: {e}")
        return {} # Return empty dict on error

    # Check if all original themes got a label, provide default if not
    for theme_id in theme_representations:
        if theme_id not in theme_labels:
            logging.warning(f"LLM did not provide a label for Theme ID {theme_id}. Using default label.")
            theme_labels[theme_id] = f"Theme {theme_id}" # Fallback label

    return theme_labels


# --- BERTopic Grouped Theme Analysis ---

def analyze_bertopic_themes_grouped(df_merged: pd.DataFrame, topic_to_theme_map: Dict[int, int], theme_representations: Dict[int, str], theme_labels: Dict[int, str], api_key: Optional[str] = None, min_count_threshold: int = 5):
    """
    Analyzes the prevalence of grouped BERTopic themes between synesthesia (True) and baseline (False) groups.

    Args:
        df_merged: DataFrame containing original dream data merged with aggregated BERTopic topic sets (output of analyzer.aggregate_topics_to_dreams).
                   Must contain 'is_synesthesia' and 'topic_id' (as a set of topic IDs) columns.
        topic_to_theme_map: Dictionary mapping original BERTopic topic IDs to theme cluster IDs.
        api_key: Optional OpenAI API key for LLM summary.
        min_count_threshold: Minimum number of dreams a theme must appear in (in either group) to be included in the results.

    Returns:
        pd.DataFrame: DataFrame containing theme prevalence comparison results, or None if analysis fails.
    """
    logging.info("Running Grouped BERTopic Theme analysis...")
    if topic_to_theme_map is None or not topic_to_theme_map:
        logging.error("Topic to theme map is missing or empty. Cannot perform grouped analysis.")
        return None
    if df_merged is None or df_merged.empty:
        logging.error("Input DataFrame df_merged is missing or empty.")
        return None
    if not all(col in df_merged.columns for col in ['is_synesthesia', 'topic_id']):
        logging.error("Input DataFrame df_merged must contain 'is_synesthesia' and 'topic_id' (set) columns.")
        return None

    # 1. Map Topics to Themes for each Dream
    logging.info("Mapping dream topics to themes...")
    def get_themes_for_dream(topic_set):
        themes = set()
        if isinstance(topic_set, set):
            for topic_id in topic_set:
                if topic_id in topic_to_theme_map:
                    themes.add(topic_to_theme_map[topic_id])
        return themes

    df_merged['theme_id'] = df_merged['topic_id'].apply(get_themes_for_dream)

    # --- Add Logging for Topic/Theme Density ---
    avg_topic_set_size_syn = df_merged[df_merged['is_synesthesia'] == True]['topic_id'].apply(len).mean()
    avg_topic_set_size_base = df_merged[df_merged['is_synesthesia'] == False]['topic_id'].apply(len).mean()
    avg_theme_set_size_syn = df_merged[df_merged['is_synesthesia'] == True]['theme_id'].apply(len).mean()
    avg_theme_set_size_base = df_merged[df_merged['is_synesthesia'] == False]['theme_id'].apply(len).mean()

    logging.info(f"Average BERTopic topic set size per dream: Synesthesia={avg_topic_set_size_syn:.2f}, Baseline={avg_topic_set_size_base:.2f}")
    logging.info(f"Average Grouped Theme set size per dream: Synesthesia={avg_theme_set_size_syn:.2f}, Baseline={avg_theme_set_size_base:.2f}")
    # --- End Logging ---


    # 2. Analyze Theme Prevalence by Group
    logging.info("Analyzing theme prevalence between is_synesthesia=True and is_synesthesia=False groups...")
    all_themes = sorted(list(set(topic_to_theme_map.values()))) # Get unique theme IDs (0-19)

    syn_true_df = df_merged[df_merged['is_synesthesia'] == True]
    syn_false_df = df_merged[df_merged['is_synesthesia'] == False]
    total_syn_true = len(syn_true_df)
    total_syn_false = len(syn_false_df)

    if total_syn_true == 0 or total_syn_false == 0:
        logging.warning("Not enough data in is_synesthesia=True or is_synesthesia=False groups for grouped theme analysis.")
        return None

    theme_prevalence = []

    for theme_id in tqdm(all_themes, desc="Comparing theme prevalence"):
        # Count how many dreams in each group contain this theme_id
        true_has_theme = syn_true_df['theme_id'].apply(lambda themes: theme_id in themes).sum()
        false_has_theme = syn_false_df['theme_id'].apply(lambda themes: theme_id in themes).sum()

        true_no_theme = total_syn_true - true_has_theme
        false_no_theme = total_syn_false - false_has_theme

        # Calculate prevalence (proportion of dreams in the group containing the theme)
        prevalence_true = true_has_theme / total_syn_true
        prevalence_false = false_has_theme / total_syn_false

        # Chi-squared test
        contingency_table = [[true_has_theme, true_no_theme], [false_has_theme, false_no_theme]]
        try:
            chi2, p, _, _ = chi2_contingency(contingency_table, correction=False)
            p_value = f"{p:.4f}"
            chi2_value = f"{chi2:.2f}"
        except ValueError as e:
            logging.debug(f"Could not perform Chi2 test for Theme ID {theme_id} (likely low counts): {e}")
            p_value = "N/A"
            chi2_value = "N/A"

        # Get the theme representation (top words) and the concise label
        theme_words_str = theme_representations.get(theme_id, "N/A")
        theme_label_str = theme_labels.get(theme_id, f"Theme {theme_id}") # Use ID as fallback

        theme_prevalence.append({
            'Theme ID': theme_id,
            'Theme Label': theme_label_str, # Add the concise label
            'Top Words': theme_words_str,
            'Prevalence Synesthesia (True)': prevalence_true,
            'Prevalence Baseline (False)': prevalence_false,
            'Chi2': chi2_value,
            'p': p_value,
            'Count Syn (True)': true_has_theme,
            'Count Base (False)': false_has_theme
        })

    prevalence_df = pd.DataFrame(theme_prevalence)

    # 3. Filter based on minimum count threshold
    initial_theme_count = len(prevalence_df)
    prevalence_df = prevalence_df[
        (prevalence_df['Count Syn (True)'] >= min_count_threshold) |
        (prevalence_df['Count Base (False)'] >= min_count_threshold)
    ]
    filtered_theme_count = len(prevalence_df)
    logging.info(f"Filtered grouped theme results from {initial_theme_count} to {filtered_theme_count} themes based on min count threshold ({min_count_threshold}) in either group.")

    # Sort the filtered DataFrame by p-value
    prevalence_df = prevalence_df.sort_values('p')

    logging.info(f"Filtered Grouped BERTopic Theme Prevalence Comparison (Top 20 significant themes by p-value):\n{prevalence_df.head(20).to_string()}")
    logging.debug(f"Full Filtered Grouped BERTopic Theme Prevalence Comparison (min_count={min_count_threshold}):\n" + prevalence_df.to_string())

    # 4. Optional LLM Summary
    if api_key:
        logging.info("Preparing data for LLM qualitative summary of grouped themes...")
        try:
            # --- Prepare data specifically for LLM summary ---
            llm_df = prevalence_df.copy()
            llm_df['p_numeric'] = pd.to_numeric(llm_df['p'], errors='coerce')
            llm_df['Total Count'] = llm_df['Count Syn (True)'] + llm_df['Count Base (False)']

            # Apply LLM-specific filters (adjust as needed)
            llm_df_filtered = llm_df[
                (llm_df['Total Count'] >= 10) &
                (llm_df['p_numeric'] < 0.2)
            ].copy()

            logging.info(f"Found {len(llm_df_filtered)} themes meeting LLM criteria (Total Count >= 10, p < 0.2).")

            if not llm_df_filtered.empty:
                # Split into overrepresented groups
                syn_overrep = llm_df_filtered[llm_df_filtered['Prevalence Synesthesia (True)'] > llm_df_filtered['Prevalence Baseline (False)']]
                base_overrep = llm_df_filtered[llm_df_filtered['Prevalence Baseline (False)'] >= llm_df_filtered['Prevalence Synesthesia (True)']]

                # Select relevant columns for the prompt, including Theme Label and Top Words
                cols_to_show = ['Theme ID', 'Theme Label', 'Top Words', 'Prevalence Synesthesia (True)', 'Prevalence Baseline (False)', 'p']
                syn_overrep_str = syn_overrep[cols_to_show].to_string(index=False)
                base_overrep_str = base_overrep[cols_to_show].to_string(index=False)

                # --- Construct LLM prompt ---
                # Use Theme Label for easier reading, keep Top Words for context
                summary_prompt = f"""Analyze the following results comparing grouped BERTopic themes between Synesthesia users (True) vs a Baseline group (False). The data shows themes filtered by statistical significance (p < 0.2) and minimum occurrence (Total Count >= 10). Each theme is represented by an ID, a concise 'Theme Label', and its top associated words.

The themes are split into two groups:
1. Themes MORE prevalent in the Synesthesia group.
2. Themes MORE prevalent in the Baseline group.

Based *only* on the provided data, provide a brief qualitative summary for EACH group separately. Describe the key themes that appear to differentiate the groups, primarily using the 'Theme Label' for interpretation, but referencing 'Top Words' if needed for clarity. Mention the 'Theme Label' (and ID) of the themes that most strongly differentiate the groups.

--- Themes More Prevalent in Synesthesia Group (True) ---
{syn_overrep_str}

--- Themes More Prevalent in Baseline Group (False) ---
{base_overrep_str}

--- Summary ---
Synesthesia Group Themes: [Your summary focusing on themes from the first list, using Theme Label and Top Words for interpretation]

Baseline Group Themes: [Your summary focusing on themes from the second list, using Theme Label and Top Words for interpretation]
"""
                # --- Call LLM ---
                try:
                    ai = AIChat(console=False, model='gpt-4o', api_key=api_key)
                    summary_response = ai(summary_prompt)
                    logging.info(f"LLM Qualitative Summary of Grouped BERTopic Theme Differences (Split by Group):\n{summary_response}")
                except ImportError:
                    logging.warning("simpleaichat not installed. Skipping LLM summary for grouped BERTopic themes.")
                except Exception as e:
                    logging.error(f"Error calling LLM for grouped BERTopic summary: {e}")
            else:
                logging.info("No grouped themes met the criteria for LLM summary after filtering.")

        except Exception as e:
            logging.error(f"Error preparing data or prompting LLM for grouped BERTopic summary: {e}")

    # 5. Save Results
    try:
        grouped_bertopic_csv_path = 'synesthesia/synesthesia_bertopic_grouped_theme_prevalence.csv'
        prevalence_df.to_csv(grouped_bertopic_csv_path, index=False)
        logging.info(f"Saved grouped BERTopic theme prevalence results to {grouped_bertopic_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save grouped BERTopic results to CSV: {e}")

    return prevalence_df


def run_analysis_workflow(args):
    """
    Core function containing the analysis workflow logic.
    Accepts parsed arguments object.
    """
    logging.info("Starting Synesthesia Dream Analysis Workflow...")

    # 1. Load Configuration
    try:
        db_url, openai_api_key = load_dotenv_config()
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return # Exit if config fails

    # 2. Establish Database Connection
    try:
        engine = get_db_engine(db_url)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return # Exit if DB connection fails

    # 3. Load Data (with Caching)
    if os.path.exists(CACHED_SYN_PATH) and os.path.exists(CACHED_BASE_PATH):
        logging.info(f"Loading data from cache files: {CACHED_SYN_PATH}, {CACHED_BASE_PATH}")
        try:
            # Load directly from pickle files
            syn_df = pd.read_pickle(CACHED_SYN_PATH)
            base_df = pd.read_pickle(CACHED_BASE_PATH)

            # Embeddings should be loaded correctly as numpy arrays, no manual parsing needed
            # Basic check:
            if not syn_df.empty and not isinstance(syn_df['embedding'].iloc[0], np.ndarray):
                 raise TypeError("Synesthesia embeddings did not load as numpy arrays from pickle.")
            if not base_df.empty and not isinstance(base_df['embedding'].iloc[0], np.ndarray):
                 raise TypeError("Baseline embeddings did not load as numpy arrays from pickle.")

            logging.info(f"Loaded {len(syn_df)} synesthesia and {len(base_df)} baseline records from cache.")

        except Exception as e:
            logging.error(f"Error loading data from cache: {e}. Will attempt to fetch from source.")
            # Force fetching from source if cache loading fails
            syn_df, base_df = None, None
            # Optionally remove potentially corrupted cache files
            if os.path.exists(CACHED_SYN_PATH): os.remove(CACHED_SYN_PATH)
            if os.path.exists(CACHED_BASE_PATH): os.remove(CACHED_BASE_PATH)

    else:
        logging.info("Cache files not found. Fetching data from source...")
        syn_df, base_df = None, None # Ensure they are None if cache doesn't exist

    # Fetch from source if cache didn't exist or failed to load
    if syn_df is None or base_df is None:
        try:
            syn_df = load_synesthesia_data()
            # Pass a copy of syn_df to avoid modification issues if loaded from cache later
            base_df = load_baseline_data_date_matched(engine, syn_df.copy())

            # Ensure data was loaded successfully before saving to cache
            if not syn_df.empty and not base_df.empty:
                 # Embeddings should already be numpy arrays from the loading functions
                 # Save to cache using pickle
                 logging.info(f"Saving fetched data to cache files: {CACHED_SYN_PATH}, {CACHED_BASE_PATH}")
                 syn_df.to_pickle(CACHED_SYN_PATH)
                 base_df.to_pickle(CACHED_BASE_PATH)
            else:
                 logging.error("Failed to load data from source. Cannot proceed.")
                 return # Exit if source loading fails

        except Exception as e:
            logging.error(f"Failed to load data from source: {e}")
            return # Exit if data loading fails

    # Final check if dataframes are valid before proceeding
    if syn_df.empty or base_df.empty:
        logging.error("One or both datasets are empty after loading attempts. Exiting.")
        return

    # 4. Prepare Data
    logging.info('Preparing data...')
    try:
        combined_df, X, y = prepare_data(syn_df, base_df)
    except ValueError as e:
        logging.error(f"Data preparation failed: {e}")
        return # Exit if preparation fails
    except Exception as e:
        logging.error(f"Unexpected error during data preparation: {e}")
        return

    if combined_df.empty or X.shape[0] == 0:
         logging.error("Dataset is empty after preparation. Exiting.")
         return

    # 5. Find Best Model using GridSearchCV
    # This replaces the manual build_pipeline and evaluate_model_cv calls
    best_pipeline = find_best_model(X, y)

    # 6. Train and Evaluate the Best Model Found on the Test Set
    logging.info("Splitting data for final evaluation of the best model...")
    # Split indices first to easily get df_train/df_test if needed later
    indices = combined_df.index
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    df_train = combined_df.loc[indices_train] # Keep df splits if needed for SHAP/analysis
    df_test = combined_df.loc[indices_test]

    logging.info(f"Training the best pipeline on {len(X_train)} samples...")
    # The best_pipeline is already configured, just fit it on the training data
    best_pipeline.fit(X_train, y_train)

    logging.info("Evaluating the best pipeline on the test set...")
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Test Set F1 (weighted): {f1:.4f}")
    logging.info(f"Test Set ROC AUC: {roc_auc:.4f}")
    logging.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred))
    logging.info("Confusion Matrix (Test Set):\n" + str(confusion_matrix(y_test, y_pred)))


    # 8. Analyze Model Scores (on the full dataset using the best pipeline)
    logging.info("Analyzing model scores on the full dataset using the best pipeline...")
    df_with_scores, _ = analyze_model_scores(combined_df.copy(), best_pipeline, X) # Use best_pipeline


    # 9. Conditional Introspection Analyses (using the best pipeline)
    if args.run_shap:
        logging.info("--- Running SHAP Analysis ---")
        # Pass the scaled training data for background if needed by explainer
        # Note: run_shap_analysis handles scaling internally now
        run_shap_analysis(best_pipeline, X_train, X_test, df_test) # Use best_pipeline
    else:
        logging.info("--- Skipping SHAP Analysis ---")

    if args.run_tfidf:
        logging.info("--- Running TF-IDF Decile Analysis ---")
        analyze_tfidf_deciles(df_with_scores)
    else:
        logging.info("--- Skipping TF-IDF Decile Analysis ---")

    if args.run_gpt_themes:
        logging.info("--- Running GPT Per-Dream Theme Tagging ---")
        if openai_api_key:
            analyze_gpt_themes(df_with_scores, openai_api_key, DEFAULT_THEMES_LIST, args.gpt_sample_size)
        else:
            logging.warning("Skipping GPT Per-Dream Theme Tagging: OPENAI_API_KEY not found.")
    else:
        logging.info("--- Skipping GPT Per-Dream Theme Tagging ---")

    if args.run_gpt_batch_themes:
        logging.info("--- Running GPT Batch Theme Analysis ---")
        if openai_api_key:
            # Function now potentially returns two dataframes
            raw_themes_df, grouped_themes_df = analyze_gpt_batch_themes(
                df_with_scores, openai_api_key, args.gpt_batch_size, args.gpt_iterations
            )
            # You can add further processing/saving of these dataframes here if needed
            # e.g., raw_themes_df.to_csv(...)
            # if not grouped_themes_df.empty: grouped_themes_df.to_csv(...)
        else:
            logging.warning("Skipping GPT Batch Theme Analysis: OPENAI_API_KEY not found.")
    else:
        logging.info("--- Skipping GPT Batch Theme Analysis ---")

    if args.run_keyword_themes:
        logging.info("--- Running Keyword Theme Analysis (using ground truth labels) ---")
        # Pass the original combined_df which contains 'is_synesthesia' and 'text'
        # and the api_key for the optional LLM summary
        analyze_keyword_themes(combined_df, api_key=openai_api_key)
    else:
        logging.info("--- Skipping Keyword Theme Analysis ---")

    if args.run_bertopic_analysis:
        logging.info("--- Running BERTopic Theme Analysis (using ground truth labels) ---")
        # Pass the original combined_df which contains 'is_synesthesia', 'text', 'id'
        # and the api_key for the optional LLM summary
        # Can also pass the threshold here if needed, e.g., min_count_threshold=args.bertopic_min_count
        # Analyze individual topics first
        bertopic_results_df, df_merged_with_topics, analyzer = analyze_bertopic_themes(combined_df, api_key=openai_api_key) # Using default threshold for now

        if bertopic_results_df is not None:
            logging.info(f"Initial BERTopic analysis returned {len(bertopic_results_df)} topics meeting threshold.")

            # --- Optional: Grouping and Grouped Analysis ---
            # --- Optional: Grouping and Grouped Analysis ---
            if args.run_bertopic_grouping and analyzer and analyzer.topic_model and df_merged_with_topics is not None:
                logging.info("--- Running BERTopic Topic Grouping ---")
                topic_to_theme_map, theme_representations = group_bertopic_topics(analyzer.topic_model)

                if topic_to_theme_map and theme_representations:
                    logging.info("--- Generating Concise Theme Labels via LLM ---")
                    theme_labels = {} # Initialize
                    if openai_api_key:
                        theme_labels = get_llm_theme_labels(theme_representations, openai_api_key)
                    else:
                        logging.warning("Skipping LLM theme labeling: OPENAI_API_KEY not found.")
                        # Create fallback labels if LLM skipped
                        theme_labels = {theme_id: f"Theme {theme_id}" for theme_id in theme_representations.keys()}

                    if theme_labels: # Check if labels were generated (or fallback created)
                        logging.info("--- Running Grouped BERTopic Theme Analysis ---")
                        # Pass the map, representations, and new labels
                        analyze_bertopic_themes_grouped(
                            df_merged_with_topics,
                            topic_to_theme_map,
                            theme_representations,
                            theme_labels, # Pass the new labels
                            api_key=openai_api_key
                        )
                    else:
                        logging.warning("Skipping grouped BERTopic theme analysis because theme label generation failed.")
                else:
                    logging.warning("Skipping grouped BERTopic theme analysis because topic grouping failed.")
            elif args.run_bertopic_grouping:
                 logging.warning("Skipping grouped BERTopic theme analysis because initial analysis failed or prerequisites missing.")
            else:
                 logging.info("--- Skipping BERTopic Topic Grouping and Grouped Analysis ---")
            # --- End Optional Grouping ---

        else:
            logging.warning("Initial BERTopic analysis did not return results. Skipping grouping.")
    else:
        logging.info("--- Skipping BERTopic Theme Analysis ---")


    logging.info("Synesthesia Dream Analysis Workflow Completed.")
    # Potential: Return results dictionary from run_analysis_workflow if needed


def main(argv: Optional[List[str]] = None):
    """
    Parses arguments and runs the analysis workflow.
    Allows programmatic invocation by passing a list of arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze Synesthesia vs Baseline dream reports using date-matched sampling.")
    parser.add_argument("--use-pca", action='store_true', help="Use PCA for dimensionality reduction.")
    parser.add_argument("--n-components", type=int, default=100, help="Number of PCA components if --use-pca is set.")
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis (requires shap library).")
    parser.add_argument("--run-tfidf", action='store_true', help="Run TF-IDF analysis on deciles (requires nltk).")
    parser.add_argument("--run-gpt-themes", action='store_true', help="Run GPT per-dream theme tagging on deciles (requires simpleaichat, openai, pydantic, and API key).")
    parser.add_argument("--gpt-sample-size", type=int, default=None, help="Sample size for GPT per-dream theme tagging per decile (optional, reduces cost/time).")
    parser.add_argument("--run-gpt-batch-themes", action='store_true', help="Run GPT batch theme analysis on deciles (requires simpleaichat, openai, pydantic, and API key).")
    parser.add_argument("--gpt-batch-size", type=int, default=30, help="Batch size for GPT batch theme analysis.")
    parser.add_argument("--gpt-iterations", type=int, default=10, help="Number of iterations for GPT batch theme analysis.")
    parser.add_argument("--run-keyword-themes", action='store_true', help="Run keyword-based theme analysis using synesthesia/syn_themes_V2.csv.")
    parser.add_argument("--run-bertopic-analysis", action='store_true', help="Run BERTopic theme analysis using pre-trained model.")
    parser.add_argument("--run-bertopic-grouping", action='store_true', help="Run BERTopic topic grouping and subsequent grouped theme analysis (requires --run-bertopic-analysis).")

    # If argv is None, parse from sys.argv, otherwise parse the provided list
    args = parser.parse_args(argv) # Pass argv here

    # Call the workflow function with the parsed arguments
    run_analysis_workflow(args)
    # Potential: Capture and return results from run_analysis_workflow


if __name__ == "__main__":
    # When run as a script, call main() without arguments
    # so it uses sys.argv by default.
    main()

"""
Usage Examples:

# Basic run (loads data, trains model, evaluates, analyzes scores)
python synesthesia/synesthesia.py

# Run with SHAP analysis
python synesthesia/synesthesia.py --run-shap

# Run with TF-IDF analysis and LLM theming of words
python synesthesia/synesthesia.py --run-tfidf

# Run with per-dream GPT theme tagging (uses default themes list)
# Requires OPENAI_API_KEY in .env
python synesthesia/synesthesia.py --run-gpt-themes

# Run per-dream tagging on a sample of 50 from top/bottom deciles
python synesthesia/synesthesia.py --run-gpt-themes --gpt-sample-size 50

# Run with GPT batch theme analysis (default: 10 iterations, batch size 30)
# Requires OPENAI_API_KEY in .env
python synesthesia/synesthesia.py --run-gpt-batch-themes

# Run batch analysis with custom parameters
python synesthesia/synesthesia.py --run-gpt-batch-themes --gpt-batch-size 20 --gpt-iterations 5

# Run everything (SHAP, TF-IDF, Per-Dream Themes, Batch Themes)
python synesthesia/synesthesia.py --run-shap --run-tfidf --run-gpt-themes --run-gpt-batch-themes

# Run with PCA (150 components) and SHAP
python synesthesia/synesthesia.py --use-pca --n-components 150 --run-shap

# Run keyword-based theme analysis
python synesthesia/synesthesia.py --run-keyword-themes

# Run everything including keyword themes
python synesthesia/synesthesia.py --run-shap --run-tfidf --run-gpt-themes --run-gpt-batch-themes --run-keyword-themes

# Run BERTopic analysis (assumes model exists)
python synesthesia/synesthesia.py --run-bertopic-analysis

# Run everything including BERTopic
python synesthesia/synesthesia.py --run-shap --run-tfidf --run-gpt-themes --run-gpt-batch-themes --run-keyword-themes --run-bertopic-analysis

# Run BERTopic analysis AND the topic grouping stage
python synesthesia/synesthesia.py --run-bertopic-analysis --run-bertopic-grouping
"""
