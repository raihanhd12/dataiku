# -*- coding: utf-8 -*-

import dataiku
from dataiku.runnables import Runnable

import os
import gzip
import zipfile
import requests
import shutil


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class MyRunnable(Runnable):
    """The base interface for a Python runnable"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        self.client = dataiku.api_client()

    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return (100, 'NONE')

    def run(self, progress_callback):

        # Retrieving parameters
        output_folder_name = self.config.get('outputName', '')
        source = self.config.get('source', '')
        if source == 'fasttext':
            text_language = self.config.get('text_language_fasttext', '')
        else:
            text_language = self.config.get('text_language_other', '')

        # Creating new Managed Folder if needed
        project = self.client.get_project(self.project_key)
        output_folder_found = False

        for folder in project.list_managed_folders():
            if output_folder_name == folder['name']:
                output_folder = project.get_managed_folder(folder['id'])
                output_folder_found = True
                break

        if not output_folder_found:
            output_folder = project.create_managed_folder(output_folder_name)

        output_folder = dataiku.Folder(output_folder.get_definition()["id"],
                                       project_key=self.project_key)

        output_folder_path = output_folder.get_path()

        #######################################
        # Downloading and extracting the data
        #######################################

        if source == 'word2vec':
            if text_language == 'english':
                url = "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz"
            else:
                raise NotImplementedError("Word2vec vectors are only available for English. Use fastText for other languages.")

            archive_name = os.path.basename(url)

            # Download archive
            r = requests.get(url, stream=True)
            with output_folder.get_writer(archive_name) as w:
                for chunk in r.iter_content(chunk_size=100000):
                    if chunk:
                        w.write(chunk)

            # Decompress in managed folder and rename
            archive_path = os.path.join(output_folder_path, archive_name)
            outfile_path = os.path.join(output_folder_path, "Word2vec_embeddings")
            with open(outfile_path, 'wb') as f_out, gzip.open(archive_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
            os.remove(archive_path)

        elif source == 'fasttext':
            if text_language == 'english':
                url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
            elif text_language == 'french':
                url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec'
            elif text_language == 'german':
                url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec'
            else:
                raise NotImplementedError(
                    "Only English, French and German languages are supported.")
            r = requests.get(url, stream=True)
            with output_folder.get_writer("fastText_embeddings") as w:
                for chunk in r.iter_content(chunk_size=100000):
                    if chunk:
                        w.write(chunk)


        elif source == 'glove':
            if text_language == 'english':
                url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
            else:
                raise NotImplementedError("GloVe vectors are only available for English. Use fastText for other languages.")

            archive_name = os.path.basename(url)

            # Download archive
            r = requests.get(url, stream=True)
            with output_folder.get_writer(archive_name) as w:
                for chunk in r.iter_content(chunk_size=100000):
                    if chunk:
                        w.write(chunk)

            file_basename = os.path.splitext(archive_name)[0]
            file_name = file_basename + '.txt'
            file_rename = "GloVe_embeddings"

            # Unzip archive into same directory
            zip_ref = zipfile.ZipFile(os.path.join(
                output_folder_path, archive_name), 'r')
            zip_ref.extractall(output_folder_path)
            zip_ref.close()

            # remove archive
            os.remove(os.path.join(output_folder_path, archive_name))
            # rename embedding file
            os.rename(os.path.join(output_folder_path, file_name), os.path.join(output_folder_path, file_rename))


        elif source == 'elmo':
            if text_language == 'english':
                import tensorflow.compat.v1 as tf
                import tensorflow_hub as hub
                tf.disable_v2_behavior()

                elmo_model_dir = os.path.join(output_folder_path, "ELMo")

                if not os.path.exists(elmo_model_dir):
                    os.makedirs(elmo_model_dir)

                # Path for saving ELMo
                os.environ["TFHUB_CACHE_DIR"] = elmo_model_dir

                # Download ELMo
                elmo_model = hub.Module(
                    "https://tfhub.dev/google/elmo/2", trainable=False)
            else:
                raise NotImplementedError(
                    "ELMo is only available for English. Use fastText for other languages.")
        else:
            raise NotImplementedError(
                "Only Word2vec, GloVe and FastText embeddings are supported.")
        return "<br><span>The model was downloaded successfuly !</span>"
