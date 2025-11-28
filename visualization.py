import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_configs import CONFIG
from model import SymphonyClassifier
import pickle
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def extract_embeddings(model, data_loader, device):
    """
    Extract penultimate embeddings from the trained model.
    
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch_data, _, _ in data_loader:
            batch_data = batch_data.to(device)
            embeddings = model.get_embeddings(batch_data, device=device)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def aggregate_to_composition_level(embeddings, composition_ids):
    """
    Aggregate chunk-level embeddings to composition-level by averaging.
    
    Args:
        embeddings: (n_chunks, embedding_dim)
        composition_ids: (n_chunks,) - which composition each chunk belongs to
    
    Returns:
        comp_embeddings: (n_compositions, embedding_dim)
        comp_ids: list of unique composition IDs
    """
    unique_comp_ids = np.unique(composition_ids)
    comp_embeddings = []
    
    for comp_id in unique_comp_ids:
        mask = composition_ids == comp_id
        comp_embedding = embeddings[mask].mean(axis=0)
        comp_embeddings.append(comp_embedding)
    
    return np.array(comp_embeddings), unique_comp_ids

def apply_dimensionality_reduction(embeddings, method='umap', n_pca_components=40, 
                                   n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply PCA followed by UMAP or t-SNE for dimensionality reduction.
    
    Args:
        embeddings: (n_samples, original_dim)
        method: 'umap' or 'tsne'
        n_pca_components: intermediate PCA dimensions
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
    
    Returns:
        embeddings_2d: (n_samples, 2)
    """
    print(f"Original embedding shape: {embeddings.shape}")
    
    # Step 1: PCA to reduce to intermediate dimensions
    pca = PCA(n_components=min(n_pca_components, embeddings.shape[0], embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"After PCA: {embeddings_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Step 2: UMAP or t-SNE to 2D
    if method == 'umap':
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, embeddings_pca.shape[0] - 1),
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
            random_state=random_state
        )
        embeddings_2d = reducer.fit_transform(embeddings_pca)
        print(f"After UMAP: {embeddings_2d.shape}")
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, embeddings_pca.shape[0] - 1),
            random_state=random_state
        )
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        print(f"After t-SNE: {embeddings_2d.shape}")
    else:
        raise ValueError("method must be 'umap' or 'tsne'")
    
    return embeddings_2d

def plot_embeddings(embeddings_2d, labels, label_names, title, filename, 
                   figsize=(12, 10), s=20, alpha=0.6):
    """
    Create a scatter plot of 2D embeddings colored by labels.
    
    Args:
        embeddings_2d: (n_samples, 2)
        labels: (n_samples,) - integer labels
        label_names: dict mapping label IDs to names
        title: plot title
        filename: output filename
        s: marker size
        alpha: marker transparency
    """
    plt.figure(figsize=figsize)
    
    # Create color map
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    colors = cm.get_cmap('tab20' if n_labels <= 20 else 'hsv')(np.linspace(0, 1, n_labels))
    
    # Plot each class
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names.get(label, f"Label {label}")
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=label_name,
            s=s,
            alpha=alpha,
            edgecolors='none'
        )
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def visualize_embeddings(folder_path, model_name, mode, output_dir='visualizations', 
                        data_split='test', method='umap'):
    """
    Main visualization pipeline.
    
    Args:
        folder_path: path to dataset folder containing .npz files
        model_name: name of the trained model (without .pth extension)
        mode: 'composer_era', 'composer', or 'era'
        output_dir: directory to save visualization plots
        data_split: 'train' or 'test'
        method: 'umap' or 'tsne'
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {data_split} data...")
    print(f"{'='*60}\n")
    
    data = np.load(os.path.join(folder_path, f"{data_split}.npz"))
    X = data["X"]
    y_composer = data["y_composer"]
    y_era = data["y_era"]
    
    # Load metadata if available
    metadata_path = "dataset_metadata.pkl"
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # For test split, we need to create composition IDs
        # Since test data is also flattened, we'll create approximate IDs based on chunk count
        # This is a limitation - ideally preprocessing should save separate metadata for train/test
        print("Warning: Using approximate composition IDs for visualization.")
        print("For best results, modify preprocessing to save composition IDs in .npz files.")
        
        # Create dummy composition IDs based on chunk groupings
        # This assumes chunks from same composition are sequential (which they should be)
        composition_ids = np.zeros(len(y_composer), dtype=np.int64)
        current_comp_id = 0
        for i in range(1, len(y_composer)):
            # Simple heuristic: if composer changes, it's likely a new composition
            if y_composer[i] != y_composer[i-1]:
                current_comp_id += 1
            composition_ids[i] = current_comp_id
        
        composer_to_id = metadata['composer_to_id']
        era_to_id = metadata['era_to_id']
    else:
        print("Warning: dataset_metadata.pkl not found. Using default mappings.")
        composition_ids = np.arange(len(y_composer)) // 50  # Rough estimate
        composer_to_id = {f"Composer_{i}": i for i in range(25)}
        era_to_id = {f"Era_{i}": i for i in range(5)}
    
    # Reverse mappings for plotting
    id_to_composer = {v: k for k, v in composer_to_id.items()}
    id_to_era = {v: k for k, v in era_to_id.items()}
    
    # Create DataLoader
    X_tensor = torch.from_numpy(X).float()
    y_composer_tensor = torch.from_numpy(y_composer).long()
    y_era_tensor = torch.from_numpy(y_era).long()
    dataset = TensorDataset(X_tensor, y_composer_tensor, y_era_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}.pth")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymphonyClassifier(
        input_size=X.shape[2],
        n_embedding=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    model_path = os.path.join(model_name + ".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Extract embeddings
    print(f"\n{'='*60}")
    print(f"Extracting embeddings...")
    print(f"{'='*60}\n")
    
    embeddings = extract_embeddings(model, data_loader, device)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    
    ##### Chunk-level visualization
    print(f"\n{'='*60}")
    print(f"Creating chunk-level visualizations ({method.upper()})...")
    print(f"{'='*60}\n")
    
    embeddings_2d_chunk = apply_dimensionality_reduction(
        embeddings, 
        method=method,
        n_pca_components=40,
        n_neighbors=15,
        min_dist=0.1
    )
    
    # Plot by composer
    plot_embeddings(
        embeddings_2d_chunk,
        y_composer,
        id_to_composer,
        f'Chunk-level Embeddings by Composer ({method.upper()})',
        os.path.join(output_dir, f'chunk_level_composer_{method}_{data_split}.png'),
        s=10,
        alpha=0.5
    )
    
    # Plot by era
    plot_embeddings(
        embeddings_2d_chunk,
        y_era,
        id_to_era,
        f'Chunk-level Embeddings by Era ({method.upper()})',
        os.path.join(output_dir, f'chunk_level_era_{method}_{data_split}.png'),
        s=10,
        alpha=0.5
    )
    
    ##### Composition-level visualization
    print(f"\n{'='*60}")
    print(f"Creating composition-level visualizations ({method.upper()})...")
    print(f"{'='*60}\n")
    
    comp_embeddings, unique_comp_ids = aggregate_to_composition_level(embeddings, composition_ids)
    print(f"Aggregated to {len(unique_comp_ids)} compositions")
    
    # Get labels for compositions
    comp_composers = []
    comp_eras = []
    for comp_id in unique_comp_ids:
        mask = composition_ids == comp_id
        comp_composers.append(y_composer[mask][0])  # Take first chunk's label
        comp_eras.append(y_era[mask][0])
    comp_composers = np.array(comp_composers)
    comp_eras = np.array(comp_eras)
    
    embeddings_2d_comp = apply_dimensionality_reduction(
        comp_embeddings,
        method=method,
        n_pca_components=min(30, comp_embeddings.shape[0] - 1),
        n_neighbors=min(10, comp_embeddings.shape[0] - 1),
        min_dist=0.1
    )
    
    # Plot by composer
    plot_embeddings(
        embeddings_2d_comp,
        comp_composers,
        id_to_composer,
        f'Composition-level Embeddings by Composer ({method.upper()})',
        os.path.join(output_dir, f'composition_level_composer_{method}_{data_split}.png'),
        s=100,
        alpha=0.7
    )
    
    # Plot by era
    plot_embeddings(
        embeddings_2d_comp,
        comp_eras,
        id_to_era,
        f'Composition-level Embeddings by Era ({method.upper()})',
        os.path.join(output_dir, f'composition_level_era_{method}_{data_split}.png'),
        s=100,
        alpha=0.7
    )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Check the '{output_dir}' folder.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python visualization.py <data_folder> <model_name> <mode> [data_split] [method]")
        print("Example: python visualization.py ML/dataset best_model composer_era test umap")
        print("\nOptional arguments:")
        print("  data_split: 'train' or 'test' (default: 'test')")
        print("  method: 'umap' or 'tsne' (default: 'umap')")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    data_split = sys.argv[4] if len(sys.argv) > 4 else 'test'
    method = sys.argv[5] if len(sys.argv) > 5 else 'umap'
    
    visualize_embeddings(data_folder, model_name, mode, data_split=data_split, method=method)