# Dreaming and Synesthesia: Differences in Symbolic Cognition and Narrative Structure

**Authors:** Emily Cook and Kyle Napierkowski

This repository contains the code and resources supplementing the academic publication titled "Dreaming and Synesthesia: Differences in Symbolic Cognition and Narrative Structure".

## Abstract

Dreams provide a unique context for examining the symbolic and associative processes that underlie human cognition. This study investigated whether individuals with synesthesia, a condition characterized by involuntary cross-modal associations, exhibit differences in dream content compared to non-synesthetes. A large corpus of Reddit dream reports (N = 2,337) was analyzed using semantic embeddings, logistic regression classification, and unsupervised topic modeling. Logistic regression achieved moderate classification performance (F1 = 0.58, ROC-AUC = 0.61). Topic modeling revealed five themes—digital distraction, Interpersonal regret, diverse worlds, non-human entities, and violent conflict—that were significantly more prevalent in synesthete dreams. No themes were found to be more prevalent among controls. These findings support the continuity hypothesis by demonstrating that stable cognitive traits shape the symbolic and thematic structure of dreams. The results also align with theories of symbolic abstraction and associative integration, suggesting that synesthesia influences the way experiences are recombined and represented during dreaming. Limitations related to sample identification and narrative salience are discussed. These findings contribute to a growing understanding of how individual cognitive architectures manifest across waking and dreaming states.

## Repository Contents

This repository includes the Python scripts used for data processing, analysis, and modeling described in the paper:

*   **`synesthesia.py`**: Main script for loading data (synesthete and baseline dream reports), preparing data (embeddings, PCA), training and evaluating classification models (Logistic Regression, Random Forest), performing hyperparameter tuning (GridSearchCV), running SHAP analysis, and conducting thematic analyses using keyword matching and BERTopic models.
*   **`bertopic_trainer.py`**: Script for training the base BERTopic model on a large corpus of dream reports, saving the model, and providing utilities for analyzing dream topics. Includes functions for loading data from a database, sentence splitting, and model persistence.
*   **`bertopic_model/`**: Directory containing the pre-trained BERTopic model (`dreams_sentence_model`) used for topic analysis in `synesthesia.py`.
*   **`requirements.txt`**: Lists the necessary Python packages to run the code.
*   **`.env` (example)**: Configuration file template for environment variables (e.g., API keys, database URLs). *Note: The actual `.env` file is not included for security reasons.*
*   **Data Files (Not Included)**: The raw dream report data (`synesthesia_dreams.csv`, baseline data) and intermediate processed files (`synesthesia_processed.pkl`, `baseline_processed.pkl`, `gridsearch_results.joblib`) are not included in this repository due to size and privacy considerations but were generated/used by the scripts.

## Methodology Overview

The analysis pipeline involved:
1.  **Data Collection**: Sourcing dream reports from Reddit (r/dreams) and identifying synesthete authors based on activity in r/Synesthesia. Date-matched controls were selected.
2.  **Feature Generation**: Creating semantic embeddings (OpenAI `text-embedding-3-large`) for each dream report.
3.  **Classification**: Training Logistic Regression and Random Forest models to distinguish between synesthete and control dreams based on embeddings. Hyperparameter tuning was performed using GridSearchCV.
4.  **Topic Modeling**: Applying a pre-trained BERTopic model (trained on a larger dream corpus using `bertopic_trainer.py`) to identify latent topics in the synesthete/control dataset.
5.  **Theme Analysis**: Hierarchically clustering the fine-grained BERTopic topics into broader themes and comparing theme prevalence between groups using Chi-squared tests.

Refer to the full paper for detailed methodology, results (including Tables 1-3), discussion, and limitations.

## How to Use

1.  Clone the repository.
2.  Set up a Python environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file based on the required variables (e.g., database connection string, OpenAI API key if needed for re-analysis).
4.  Obtain the necessary data files (dream reports, pre-trained models if not using the included one).
5.  Run the scripts (`bertopic_trainer.py` to train a new model, `synesthesia.py` for the main analysis workflow). *Note: Running the full analysis requires access to the original datasets and potentially significant computational resources.*

This code is provided for transparency and reproducibility of the methods described in the publication.
