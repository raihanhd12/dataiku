# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='[Sentence Embedding Ranking] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ============================================
# 1. Read Input Datasets and Specify Columns
# ============================================
recipe_config = get_recipe_config()

baseline_text_column = recipe_config.get("baseline_text_column")
test_text_column = recipe_config.get("test_text_column")
top_n = int(recipe_config.get("top_n", 10))
model_filename = recipe_config.get("model_filename", "Word2vec_embeddings")

# Retrieve input dataset names from recipe roles
baseline_dataset_name = get_input_names_for_role("baseline_dataset")[0]
test_dataset_name = get_input_names_for_role("test_dataset")[0]

logger.info("Loading Baseline dataset: %s", baseline_dataset_name)
df_baseline = dataiku.Dataset(baseline_dataset_name).get_dataframe()

logger.info("Loading Test dataset: %s", test_dataset_name)
df_test = dataiku.Dataset(test_dataset_name).get_dataframe()

# Extract text columns; ensure strings and remove extra spaces
base_texts = df_baseline[baseline_text_column].astype(str).str.strip()
test_texts = df_test[test_text_column].astype(str).str.strip()

logger.info("Number of rows in Baseline dataset: %d", len(base_texts))
logger.info("Number of rows in Test dataset: %d", len(test_texts))

# ============================================
# 2. Load Pre-trained Model from Dataiku Folder
# ============================================
embedding_folder_name = get_input_names_for_role("embedding_folder")[0]
folder = dataiku.Folder(embedding_folder_name)
local_folder_path = folder.get_path()
model_file = os.path.join(local_folder_path, model_filename)

logger.info("Loading pre-trained model from: %s", model_file)
model = KeyedVectors.load_word2vec_format(model_file, binary=True)

# ============================================
# 3. Function to Generate Sentence Embedding
# ============================================
def get_sentence_embedding(text, model):
    """
    Generate an embedding for a given text by:
      - Tokenizing the text (splitting by whitespace)
      - Retrieving embeddings for each token (using the token as-is, lowercase, or uppercase)
      - Returning the mean of all found token embeddings.
    If no token embeddings are found, returns a zero vector.
    """
    tokens = text.split()
    vecs = []
    for token in tokens:
        if token in model.key_to_index:
            vecs.append(model[token])
        elif token.lower() in model.key_to_index:
            vecs.append(model[token.lower()])
        elif token.upper() in model.key_to_index:
            vecs.append(model[token.upper()])
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(vecs, axis=0)

# Compute embeddings for each text in the Baseline and Test datasets.
logger.info("Computing embeddings for Baseline texts...")
base_embeddings = base_texts.apply(lambda x: get_sentence_embedding(x, model))
logger.info("Computing embeddings for Test texts...")
test_embeddings = test_texts.apply(lambda x: get_sentence_embedding(x, model))

# ============================================
# 4. Compute Cosine Similarity & Distance, Retrieve Top N Closest
# ============================================
results = []

for i, base_text in enumerate(base_texts):
    base_vec = base_embeddings.iloc[i]
    similarity_list = []
    for j, test_text in enumerate(test_texts):
        test_vec = test_embeddings.iloc[j]
        # Compute cosine similarity and derive cosine distance
        sim = cosine_similarity(base_vec.reshape(1, -1), test_vec.reshape(1, -1))[0, 0]
        dist = 1 - sim  # Lower distance indicates higher similarity
        similarity_list.append({
            "baseline_text": base_text,
            "test_text": test_text,
            "cosine_similarity": sim,
            "cosine_distance": dist
        })
    
    # Sort pairs by cosine distance (ascending order)
    similarity_list_sorted = sorted(similarity_list, key=lambda x: x["cosine_distance"])
    
    # Retrieve the top N pairs for the current baseline text
    top_entries = similarity_list_sorted[:top_n]
    for rank, entry in enumerate(top_entries, start=1):
        entry["rank"] = rank  # Rank 1 means the most similar pair
        results.append(entry)

# ============================================
# 5. Create Output DataFrame and Write to Dataiku Dataset
# ============================================
output_df = pd.DataFrame(results)
logger.info("Total output pairs (top %d per baseline): %d", top_n, len(output_df))

output_dataset_name = get_output_names_for_role("output_dataset")[0]
dataiku.Dataset(output_dataset_name).write_with_schema(output_df)
logger.info("Done.")
