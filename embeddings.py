import os
import re
import json
import requests
import pandas as pd
from tqdm import tqdm 

DATASET = "alamios/OpenSubtitles-v2018"
CONFIG = "en-es"
OUT_DIR = "opensubtitles_en_es"
os.makedirs(OUT_DIR, exist_ok=True)


def list_parquet_files(dataset: str, config: str, split: str = "train"):
    """Mirrors: curl .../api/datasets/{dataset}/parquet/{config}/{split}"""
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()  # list of parquet file URLs


def download_parquet_files(dataset: str, config: str, split: str = "train"):
    urls = list_parquet_files(dataset, config, split)
    local_paths = []
    for i, url in enumerate(urls):
        local_path = os.path.join(OUT_DIR, f"{config}_{split}_{i}.parquet")
        if not os.path.exists(local_path):
            print(f"Downloading shard {i+1}/{len(urls)}: {url}")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        local_paths.append(local_path)
    return local_paths


# Fallback if you only want a small sample instead of the full parquet dump:
def fetch_rows_via_api(dataset: str, config: str, split: str = "train",
                        offset: int = 0, length: int = 100):
    """Mirrors the /rows API from your screenshot."""
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={requests.utils.quote(dataset, safe='')}"
        f"&config={config}&split={split}&offset={offset}&length={length}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["rows"]


def build_monolingual_corpora(parquet_paths, en_path, es_path, max_rows=None):
    """Reads translation pairs and writes one sentence per line into
    separate EN and ES text files. The two files stay fully independent
    (same line count, but that's just incidental - we never mix them)."""
    seen = 0
    with open(en_path, "w", encoding="utf-8") as f_en, \
         open(es_path, "w", encoding="utf-8") as f_es:
        for path in parquet_paths:
            # NOTE: depending on the parquet engine (pyarrow vs fastparquet)
            # and how the dataset author wrote the file, the "translation"
            # struct column may show up either as:
            #   (a) a single nested column "translation" -> {"en": ..., "es": ...}
            #   (b) two already-flattened columns "translation.en" / "translation.es"
            # We detect which one is present and handle both.
            try:
                df = pd.read_parquet(path, columns=["translation.en", "translation.es"])
                en_col, es_col = "translation.en", "translation.es"
            except (ValueError, KeyError):
                df = pd.read_parquet(path, columns=["translation"])
                # expand the nested dict column into two plain columns
                df["translation.en"] = df["translation"].apply(lambda r: (r or {}).get("en"))
                df["translation.es"] = df["translation"].apply(lambda r: (r or {}).get("es"))
                en_col, es_col = "translation.en", "translation.es"

            for en_text, es_text in tqdm(
                zip(df[en_col], df[es_col]),
                total=len(df),
                desc=f"Processing {os.path.basename(path)}",
            ):
                en_text = (en_text or "").strip()
                es_text = (es_text or "").strip()
                if en_text:
                    f_en.write(en_text.replace("\n", " ") + "\n")
                if es_text:
                    f_es.write(es_text.replace("\n", " ") + "\n")
                seen += 1
                if max_rows and seen >= max_rows:
                    return

# All above is data processing. 

TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)  # keeps accented Spanish letters


def tokenize_line(line: str):
    return TOKEN_RE.findall(line.lower())


class LineSentenceTokenized:
    """Streams a text file and yields tokenized sentences, so you never
    have to load the whole corpus into memory (gensim-friendly)."""
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = tokenize_line(line)
                if tokens:
                    yield tokens




def train_word2vec(corpus_path, model_out_path, vector_size=300, window=5,
                    min_count=5, epochs=5, sg=1, workers=4):
    from gensim.models import Word2Vec 
    sentences = LineSentenceTokenized(corpus_path)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size, # no. of channels 
        window=window, # gives a context window- so the no. of words it looks for in for the skipgram. looks for 'window' number of words before and 'window' number of words afterwards
        min_count=min_count, # remove words that appear less than 'min_count' times 
        sg=sg,          # training style, skipgram or CBOW 
        workers=workers, # how many parallel operations you want, depends on no. of cores, this codespace has 4, so we will be doing 4 parallel operations 
        epochs=epochs, # iterations!
    )
    model.save(model_out_path)
    print(f"Saved model -> {model_out_path} (vocab size: {len(model.wv)})")
    return model


if __name__ == "__main__":
    # --- Option A: full dataset via parquet (recommended for real training) ---
    parquet_paths = download_parquet_files(DATASET, CONFIG, split="train")

    en_corpus = os.path.join(OUT_DIR, "corpus_en.txt")
    es_corpus = os.path.join(OUT_DIR, "corpus_es.txt")

    # Set max_rows=None to use the FULL en-es split (~61M rows - large!).
    # Start with a cap (e.g. 2_000_000) while testing your pipeline.
    build_monolingual_corpora(parquet_paths, en_corpus, es_corpus, max_rows=2_000_000)

    # --- Train independently: no shared vocabulary, no shared training ---
    en_model = train_word2vec(en_corpus, os.path.join(OUT_DIR, "word2vec_en.model"))
    es_model = train_word2vec(es_corpus, os.path.join(OUT_DIR, "word2vec_es.model"))

    # Quick sanity checks
    print("EN nearest to 'king':", en_model.wv.most_similar("king", topn=5))
    print("ES nearest to 'rey':", es_model.wv.most_similar("rey", topn=5))
