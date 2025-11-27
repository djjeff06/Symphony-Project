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
