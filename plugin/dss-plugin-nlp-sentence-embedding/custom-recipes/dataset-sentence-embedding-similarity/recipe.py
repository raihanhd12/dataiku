# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
import pandas as pd
import numpy as np
import os
import logging
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

# Setup logging
FORMAT = '[SENTENCE EMBEDDING] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

##################################
# Fungsi untuk load model embedding
##################################
def load_pretrained_model(folder_path, embedding_is_custom=False):
    """
    Memuat model embedding dari folder managed.
    Asumsi: model disimpan dalam file "Word2vec_embeddings" dengan format binary.
    """
    model_file = os.path.join(folder_path, "Word2vec_embeddings")
    logger.info("Loading model from: %s", model_file)
    model = KeyedVectors.load_word2vec_format(model_file, binary=True)
    return model

##################################
# Fungsi untuk menambahkan prefix ke dictionary
##################################
def add_prefix(d, prefix):
    """
    Mengembalikan dictionary baru di mana setiap key ditambahkan prefix.
    """
    return {f"{prefix}{k}": v for k, v in d.items()}

##################################
# Fungsi untuk Compute Sentence Embeddings
##################################
def get_sentence_embedding(text, model):
    """
    Menghasilkan embedding untuk satu kalimat dengan cara:
      - Tokenisasi (split berdasarkan spasi)
      - Mencari embedding untuk tiap token (mencoba token asli, lowercase, dan uppercase)
      - Jika token ditemukan, kembalikan rata-rata embedding; jika tidak, kembalikan vektor nol.
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

def compute_embeddings(texts, model, aggregation_method='simple_average', smoothing_parameter=0.001, npc=1):
    """
    Menghitung embedding untuk sekumpulan teks.
      - Jika 'simple_average': kembalikan rata-rata embedding tiap token.
      - Jika 'SIF': hitung rata-rata berbobot dan hilangkan komponen utama.
    """
    if aggregation_method == 'simple_average':
        return [get_sentence_embedding(str(text), model) for text in texts]
    elif aggregation_method == 'SIF':
        embeddings = [get_sentence_embedding(str(text), model) for text in texts]
        X = np.vstack(embeddings)
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        pc = svd.components_
        embeddings_sif = []
        for emb in embeddings:
            for comp in pc:
                emb = emb - np.dot(comp, emb) * comp
            embeddings_sif.append(emb)
        return embeddings_sif
    else:
        raise ValueError("Unsupported aggregation method: {}".format(aggregation_method))

##################################
# Setup Fungsi Jarak (Distance Function)
##################################
def get_distance_function(distance_metric):
    if distance_metric == "cosine":
        return cosine  # cosine() mengembalikan cosine distance (1 - cosine similarity)
    elif distance_metric == "euclidean":
        return euclidean
    elif distance_metric == "absolute":
        return lambda x, y: np.linalg.norm(np.array(x) - np.array(y), ord=1)
    elif distance_metric == "wasserstein":
        return wasserstein_distance
    else:
        raise ValueError("Unsupported distance metric: {}".format(distance_metric))

##################################
# MAIN: Recipe Plugin
##################################
logger.info("Starting recipe plugin...")

# --- Ambil input dan output ---
baseline_dataset_name = get_input_names_for_role("baseline_dataset")[0]
test_dataset_name = get_input_names_for_role("test_dataset")[0]
embedding_folder_name = get_input_names_for_role("embedding_folder")[0]
output_dataset_name = get_output_names_for_role("output_dataset")[0]

logger.info("Loading baseline dataset: %s", baseline_dataset_name)
df_baseline = dataiku.Dataset(baseline_dataset_name).get_dataframe()

logger.info("Loading test dataset: %s", test_dataset_name)
df_test = dataiku.Dataset(test_dataset_name).get_dataframe()

# --- Ambil parameter dari recipe config ---
recipe_config = get_recipe_config()

baseline_text_column = recipe_config.get("baseline_text_column")
test_text_column = recipe_config.get("test_text_column")
distance_metric = recipe_config.get("distance", "cosine")
aggregation_method = recipe_config.get("aggregation_method", "simple_average")
top_n = int(recipe_config.get("top_n", 10))
embedding_is_custom = recipe_config.get("embedding_is_custom", False)

if aggregation_method == "SIF":
    advanced_settings = recipe_config.get("advanced_settings", False)
    if advanced_settings:
        smoothing_parameter = float(recipe_config.get("smoothing_parameter", 0.001))
        npc = int(recipe_config.get("n_principal_components", 1))
    else:
        smoothing_parameter = 0.001
        npc = 1
else:
    smoothing_parameter, npc = None, None

baseline_prefix = recipe_config.get("baseline_prefix", "baselinePrefix_")
test_prefix = recipe_config.get("test_prefix", "testPrefix_")

# --- Load embedding model ---
logger.info("Loading embedding model from folder: %s", embedding_folder_name)
folder_path = dataiku.Folder(embedding_folder_name).get_path()
model = load_pretrained_model(folder_path, embedding_is_custom)

# --- Compute embeddings untuk masing-masing kolom ---
logger.info("Computing sentence embeddings for baseline dataset column: %s", baseline_text_column)
baseline_texts = df_baseline[baseline_text_column].astype(str).str.strip().tolist()
baseline_embeddings = compute_embeddings(baseline_texts, model, aggregation_method, smoothing_parameter, npc)

logger.info("Computing sentence embeddings for test dataset column: %s", test_text_column)
test_texts = df_test[test_text_column].astype(str).str.strip().tolist()
test_embeddings = compute_embeddings(test_texts, model, aggregation_method, smoothing_parameter, npc)

# --- Setup distance function ---
distance_function = get_distance_function(distance_metric)

# --- Hitung pairwise distance, ranking, dan gabungkan seluruh kolom ---
logger.info("Computing pairwise distances and ranking top %d matches per baseline text...", top_n)
results = []
for i, (base_text, base_emb) in enumerate(zip(baseline_texts, baseline_embeddings)):
    # Ambil seluruh kolom dari baris baseline dan tambahkan prefix
    baseline_row = df_baseline.iloc[i].to_dict()
    baseline_row_prefixed = add_prefix(baseline_row, baseline_prefix)
    
    similarity_list = []
    for j, (test_text, test_emb) in enumerate(zip(test_texts, test_embeddings)):
        test_row = df_test.iloc[j].to_dict()
        test_row_prefixed = add_prefix(test_row, test_prefix)
        
        # Hitung jarak (distance); jika terdapat NaN, set sebagai NaN
        if np.isnan(base_emb).any() or np.isnan(test_emb).any():
            dist = np.nan
        else:
            dist = distance_function(base_emb, test_emb)
        
        # Gabungkan dictionary baseline dan test, serta tambahkan field computed
        pair_dict = {}
        pair_dict.update(baseline_row_prefixed)
        pair_dict.update(test_row_prefixed)
        pair_dict[f"{baseline_prefix}test_distance"] = dist
        # if distance_metric == "cosine":
        #     # cosine_similarity = 1 - cosine_distance
        #     pair_dict[f"{baseline_prefix}test_cosine_similarity"] = 1 - dist
        pair_dict[f"{baseline_prefix}index_baseline"] = i
        pair_dict[f"{test_prefix}index_test"] = j
        
        similarity_list.append(pair_dict)
    
    # Urutkan pasangan berdasarkan distance (ascending)
    similarity_list_sorted = sorted(
        similarity_list,
        key=lambda x: x[f"{baseline_prefix}test_distance"] if not np.isnan(x[f"{baseline_prefix}test_distance"]) else np.inf
    )
    
    # Ambil top_n pasangan teratas dan tambahkan ranking
    for rank, entry in enumerate(similarity_list_sorted[:top_n], start=1):
        entry[f"{baseline_prefix}test_rank"] = rank
        results.append(entry)

logger.info("Finished computing similarity for %d baseline texts.", len(baseline_texts))
output_df = pd.DataFrame(results)
logger.info("Total rows in output: %d", len(output_df))

# --- Tulis output ke dataset ---
dataiku.Dataset(output_dataset_name).write_with_schema(output_df)
logger.info("Output written to dataset: %s", output_dataset_name)
