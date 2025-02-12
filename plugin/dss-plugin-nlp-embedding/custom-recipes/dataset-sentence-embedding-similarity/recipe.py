# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import *
from commons import load_pretrained_model

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine, euclidean
import logging

# Setup logging
FORMAT = '[DATASET SENTENCE EMBEDDING] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

##################################
# Input Data
##################################

# Retrieve the Baseline and Test datasets
baseline_dataset_name = get_input_names_for_role('baseline_dataset')[0]
test_dataset_name = get_input_names_for_role('test_dataset')[0]

logger.info("Loading Baseline dataset: %s", baseline_dataset_name)
df_baseline = dataiku.Dataset(baseline_dataset_name).get_dataframe()

logger.info("Loading Test dataset: %s", test_dataset_name)
df_test = dataiku.Dataset(test_dataset_name).get_dataframe()

# Retrieve the embedding folder
embedding_folder_name = get_input_names_for_role('embedding_folder')[0]
folder_path = dataiku.Folder(embedding_folder_name).get_path()

##################################
# Parameters
##################################

recipe_config = get_recipe_config()

# Select the text column for each dataset
baseline_text_col = recipe_config.get('text_column_baseline', None)
if baseline_text_col is None:
    raise ValueError("You have not selected a text column for the Baseline Dataset.")

test_text_col = recipe_config.get('text_column_test', None)
if test_text_col is None:
    raise ValueError("You have not selected a text column for the Test Dataset.")

# Distance parameter
distance = recipe_config.get('distance', None)
if distance is None:
    raise ValueError("You have not selected a distance method.")

# Aggregation method and custom embedding parameter
embedding_is_custom = recipe_config.get('embedding_is_custom', False)
aggregation_method = recipe_config.get('aggregation_method', None)
if aggregation_method is None:
    raise ValueError("You have not selected an aggregation method.")

if aggregation_method == 'simple_average':
    smoothing_parameter, npc = None, None
elif aggregation_method == 'SIF':
    advanced_settings = recipe_config.get('advanced_settings', False)
    if advanced_settings:
        smoothing_parameter = float(recipe_config.get('smoothing_parameter'))
        npc = int(recipe_config.get('n_principal_components'))
    else:
        smoothing_parameter = 0.001
        npc = 1

# Parameter for top N ranking
top_n = int(recipe_config.get('top_n', 10))

##################################
# Loading Embedding Model
##################################

logger.info("Loading pre-trained model from folder: %s", folder_path)
model = load_pretrained_model(folder_path, embedding_is_custom)

##################################
# Compute Sentence Embeddings
##################################

logger.info("Computing embeddings for the Baseline Dataset...")
baseline_texts = df_baseline[baseline_text_col].astype(str).tolist()
if aggregation_method == 'simple_average':
    baseline_embeddings = model.get_sentence_embedding(baseline_texts)
else:
    baseline_embeddings = model.get_weighted_sentence_embedding(baseline_texts, smoothing_parameter, npc)

logger.info("Computing embeddings for the Test Dataset...")
test_texts = df_test[test_text_col].astype(str).tolist()
if aggregation_method == 'simple_average':
    test_embeddings = model.get_sentence_embedding(test_texts)
else:
    test_embeddings = model.get_weighted_sentence_embedding(test_texts, smoothing_parameter, npc)

##################################
# Define Distance Function
##################################

if distance == "cosine":
    distance_function = cosine
elif distance == "euclidean":
    distance_function = euclidean
elif distance == "absolute":
    def distance_function(x, y):
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x - y, ord=1)
elif distance == "wasserstein":
    distance_function = wasserstein_distance
else:
    raise ValueError("The selected distance method is not supported.")

##################################
# Compute Pairwise Similarity
##################################

logger.info("Computing similarity for each Baseline-Test pair...")

results = []
for i, emb_baseline in enumerate(baseline_embeddings):
    for j, emb_test in enumerate(test_embeddings):
        # Ensure the embedding is valid (contains no NaN)
        if np.sum(np.isnan(emb_baseline)) == 0 and np.sum(np.isnan(emb_test)) == 0:
            d = distance_function(emb_baseline, emb_test)
        else:
            d = np.nan
        results.append({
            "baseline_code": baseline_texts[i],
            "test_code": test_texts[j],
            "similarity_distance": d
        })

##################################
# Sorting and Ranking
##################################

# Sort pairs by similarity_distance (smaller values indicate higher similarity)
results_sorted = sorted(results, key=lambda x: x["similarity_distance"] if not np.isnan(x["similarity_distance"]) else np.inf)

# Take the top N pairs
top_results = results_sorted[:top_n]

# Add ranking to the top pairs
for rank, item in enumerate(top_results, start=1):
    item["ranking"] = rank

##################################
# Writing Output
##################################

logger.info("Writing output with top %d pairs", top_n)
df_out = pd.DataFrame(top_results)

output_dataset_name = get_output_names_for_role('output_dataset')[0]
dataiku.Dataset(output_dataset_name).write_with_schema(df_out)

logger.info("Done.")
