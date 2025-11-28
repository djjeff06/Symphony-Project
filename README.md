# Team 27

## Project Overview

This project is part of our deliverables for the course of Machine Learning (CSIE5043) at National Taiwan University.

## Hardware & System Requirements

- GPU: CUDA-enabled GPU with at least 4GB VRAM (required for training)
- CPU: Any modern multi-core processor
- Python: 3.10.12 (ensure dependencies are installed)

> Note: For inference/testing, a GPU is recommended but not strictly required.

## Instructions

1. Download the dataset: https://www.space.ntu.edu.tw/navigate/a/#/s/7BDECE88F1184391AF062D5383E3B5686BL

2. Install project dependencies: 
   'pip install -r requirements.txt' 
   or ensure all packages listed in requirements.txt are installed in your environment.
   
3. Training a new model using the extracted dataset:
   'python training.py data_path model_name mode'

   <p>Note:<br>
   'data_path': path containing training file (e.g., project/dataset)<br>
   'model_name': name of the trained model (no file extension needed) (e.g., best_model)<br>
   'mode': 'composer_era', 'composer', 'era' -> choose which label(s) to predict
   </p>
   
4. For hyperparameter tuning, change the values in model_configs.py.

5. To run inference on test data:
   'python inference.py data_path model_name mode'
   
   <p>Note:<br>
   'data_path': path containing test file (e.g., project/dataset)<br>
   'model_name': name of the trained model (no file extension needed, include path if needed) (e.g., best_model, models/best_model)<br>
   'mode': 'composer_era', 'composer', 'era' -> choose which label(s) to predict
   </p>

6. To visualize embeddings from a trained model:
   'python visualization.py data_path model_name mode [data_split] [method]'
   
   <p>Note:<br>
   'data_path': path containing data file (e.g., project/dataset)<br>
   'model_name': name of the trained model (no file extension needed) (e.g., best_model)<br>
   'mode': 'composer_era', 'composer', 'era' -> must match the model's training mode<br>
   'data_split': (optional) 'train' or 'test' (default: 'test')<br>
   'method': (optional) 'umap' or 'tsne' (default: 'umap')<br>
   </p>
   
   Examples:
   - 'python visualization.py project/dataset best_model composer_era test umap'
   - 'python visualization.py project/dataset best_model composer train tsne'
   
   The script will generate visualization plots in the 'visualizations/' folder:
   - Chunk-level embeddings colored by composer and era
   - Composition-level embeddings colored by composer and era
   
   <p>Note: For best results, ensure you've run preprocessing with the updated 
   'standardize_to_np_with_metadata' function to generate dataset_metadata.pkl</p>