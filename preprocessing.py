import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pickle
import sys
from sklearn.model_selection import train_test_split

def audio_to_tokens(
    path,
    sr=22050,
    chunk_seconds=10,
    trim_first_seconds=20,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
    patch_mel=16,
    patch_time=16
):
    """
    Convert an audio file to a list of token chunks.
    Each chunk becomes: [num_tokens, patch_mel * patch_time]
    """

    # ------------------------
    # 1. Load audio
    # ------------------------
    y, sr = librosa.load(path, sr=sr, mono=True)

    # ------------------------
    # 2. Remove first N seconds (silence, audience noise)
    # ------------------------
    trim_samples = trim_first_seconds * sr
    if len(y) > trim_samples:
        y = y[trim_samples:]

    # ------------------------
    # 3. Split into fixed-length chunks
    # ------------------------
    chunk_length = chunk_seconds * sr
    num_chunks = len(y) // chunk_length

    tokens_per_chunk = []

    for i in range(num_chunks):
        chunk = y[i * chunk_length : (i + 1) * chunk_length]

        # ------------------------
        # 4. Mel-spectrogram
        # ------------------------
        S = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # ------------------------
        # 5. Patchify
        # ------------------------
        H = (S_db.shape[0] // patch_mel) * patch_mel
        W = (S_db.shape[1] // patch_time) * patch_time
        S_crop = S_db[:H, :W]

        patches = S_crop.reshape(
            H // patch_mel, patch_mel,
            W // patch_time, patch_time
        )
        patches = patches.transpose(0, 2, 1, 3)

        tokens = patches.reshape(-1, patch_mel * patch_time)

        tokens_per_chunk.append(tokens)

    return tokens_per_chunk

def load_dataset(root_dir="Symphonies"):
    """
    Walk through:
        Symphonies / Era / Composer / file.(mp3|wav)
    
    Returns a list of dicts:
        {
            'tokens': np.ndarray [num_chunks, num_tokens, token_dim],
            'era': era_label,
            'composer': composer_label,
            'file': filename
        }
    """
    dataset = []
    valid_ext = (".mp3", ".wav")

    for era in os.listdir(root_dir):
        era_path = os.path.join(root_dir, era)
        if not os.path.isdir(era_path):
            continue

        for composer in os.listdir(era_path):
            comp_path = os.path.join(era_path, composer)
            if not os.path.isdir(comp_path):
                continue

            for fname in os.listdir(comp_path):
                if not fname.lower().endswith(valid_ext):
                    continue

                fpath = os.path.join(comp_path, fname)

                try:
                    tokens = audio_to_tokens(fpath)
                    if len(tokens) == 0:
                        print(f"[WARNING] File too short, skipping: {fpath}")
                        continue

                    dataset.append({
                        "tokens": tokens,        # list of [num_tokens, 256]
                        "era": era,              # label 1
                        "composer": composer,    # label 2
                        "file": fname
                    })

                    print(f"Loaded {fname} ({era}/{composer}) → {len(tokens)} chunks")

                except Exception as e:
                    print(f"[ERROR] Failed on {fpath}: {repr(e)}")

    return dataset

def cap_chunks_per_composition(dataset, max_chunks_per_comp=200, seed=15):
    random.seed(seed)
    new_dataset = []
    for item in dataset:
        tokens_list = item["tokens"]
        n = len(tokens_list)
        if n <= max_chunks_per_comp:
            new_dataset.append(item)
        else:
            sel_idx = sorted(random.sample(range(n), max_chunks_per_comp))
            new_item = item.copy()
            new_item["tokens"] = [tokens_list[i] for i in sel_idx]
            new_dataset.append(new_item)
    return new_dataset

def standardize_to_np(dataset):
    # Extract unique classes
    all_composers = sorted(list({item["composer"] for item in dataset}))
    all_eras = sorted(list({item["era"] for item in dataset}))

    composer_to_id = {c: i for i, c in enumerate(all_composers)}
    era_to_id = {e: i for i, e in enumerate(all_eras)}

    print("Composer IDs:", composer_to_id)
    print("Era IDs:", era_to_id)

    X_chunks = []
    y_composer = []
    y_era = []

    # Process sequences
    for item in dataset:
        comp_id = composer_to_id[item["composer"]]
        era_id = era_to_id[item["era"]]

        for chunk in item["tokens"]:
            chunk = chunk.astype(np.float32)

            # Standardize per chunk
            mean = chunk.mean()
            std = chunk.std() + 1e-9
            chunk = (chunk - mean) / std

            X_chunks.append(chunk)
            y_composer.append(comp_id)
            y_era.append(era_id)

    # Convert list to numpy (no padding)
    X = np.stack(X_chunks)   # shape: (n_chunks, seq_len, token_dim)

    return (
        X,
        np.array(y_composer, dtype=np.int64),
        np.array(y_era, dtype=np.int64),
        composer_to_id,
        era_to_id
    )
    
def standardize_to_np_with_metadata(dataset):
    """
    Extended version that also tracks composition IDs for each chunk (for better visualization).
    """
    # Extract unique classes
    all_composers = sorted(list({item["composer"] for item in dataset}))
    all_eras = sorted(list({item["era"] for item in dataset}))

    composer_to_id = {c: i for i, c in enumerate(all_composers)}
    era_to_id = {e: i for i, e in enumerate(all_eras)}

    print("Composer IDs:", composer_to_id)
    print("Era IDs:", era_to_id)

    X_chunks = []
    y_composer = []
    y_era = []
    composition_ids = []  # Tracks which composition each chunk belongs to
    composition_names = []  # Tracks composition filenames
    
    # Assign unique composition ID to each item
    for comp_id, item in enumerate(dataset):
        comp_label = composer_to_id[item["composer"]]
        era_label = era_to_id[item["era"]]
        comp_name = item["file"]

        for chunk in item["tokens"]:
            chunk = chunk.astype(np.float32)

            # Standardize per chunk
            mean = chunk.mean()
            std = chunk.std() + 1e-9
            chunk = (chunk - mean) / std

            X_chunks.append(chunk)
            y_composer.append(comp_label)
            y_era.append(era_label)
            composition_ids.append(comp_id)  # New
            composition_names.append(comp_name)  # New

    # Convert to numpy
    X = np.stack(X_chunks)

    return (
        X,
        np.array(y_composer, dtype=np.int64),
        np.array(y_era, dtype=np.int64),
        np.array(composition_ids, dtype=np.int64),  # New
        composition_names,  # New
        composer_to_id,
        era_to_id
    )

def preprocess():
    dataset = load_dataset("Symphonies")
    dataset_capped = cap_chunks_per_composition(dataset, max_chunks_per_comp=60)
    '''
    with open("dataset_capped.pkl", "wb") as f:
        pickle.dump(dataset_capped, f)
    '''
    #X_pad, y_composer, y_era, composer_to_id, era_to_id = standardize_to_np(dataset_capped)
    X_pad, y_composer, y_era, comp_ids, comp_names, composer_to_id, era_to_id = standardize_to_np_with_metadata(dataset_capped)

    X_train, X_test, y_composer_train, y_composer_test, y_era_train, y_era_test = train_test_split(
        X_pad, 
        y_composer, 
        y_era, 
        test_size=0.2,        # 20% test
        random_state=15, 
        stratify=y_composer    # stratify by composer
    )

    # Create folder if it doesn't exist
    save_dir = "train-test"
    os.makedirs(save_dir, exist_ok=True)

    # Save training set
    np.savez_compressed(
        os.path.join(save_dir, "train.npz"),
        X=X_train,
        y_composer=y_composer_train,
        y_era=y_era_train
    )

    # Save test set
    np.savez_compressed(
        os.path.join(save_dir, "test.npz"),
        X=X_test,
        y_composer=y_composer_test,
        y_era=y_era_test
    )

    print(f"Saved train and test datasets to '{save_dir}' folder.")
    
    # Save metadata for visualization
    metadata = {
        'composition_ids': comp_ids,
        'composition_names': comp_names,
        'composer_to_id': composer_to_id,
        'era_to_id': era_to_id
    }
    
    with open("dataset_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("Saved metadata to dataset_metadata.pkl")


if __name__ == "__main__":
    preprocess()
