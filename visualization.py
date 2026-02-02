""" Library for dta visualization using t-SNE, PCA and UMAP"""
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import numpy as np
import matplotlib.pyplot as plt



def tsne_classification_pipeline(X, y, class_names=None):
    """
    Complete t-SNE visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Optional PCA preprocessing for high-dimensional data
    if X.shape[1] > 50:
        print(f"High dimensional data ({X.shape[1]} features). Applying PCA first...")
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 3: Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,  # Typical values: 5-50
        max_iter=1000,  # Increase for better convergence
        learning_rate=200,  # Auto or 200 is often good
        early_exaggeration=12,  # Default is 12
        metric='euclidean'  # Can also try 'manhattan', 'cosine'
    )

    X_tsne = tsne.fit_transform(X_scaled)

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('t-SNE Visualization of Classification Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_tsne


def pca_classification_pipeline(X, y, class_names=None):
    """
    Complete PCA visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA

    print(f"High dimensional data ({X.shape[1]} features). Applying PCA ...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('PCA Visualization of Classification Data')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_pca

def umap_classification_pipeline(X, y, class_names=None, X2=None, y2=None, return_umap=False):
    """
    Complete UMAP visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply UMAP

    print(f"High dimensional data ({X.shape[1]} features). Applying UMAP...")
    umap_viz = umap.UMAP(n_components=2,
                         n_neighbors=10,  # Number of neighbors (5-50)
                         min_dist=0.05,  # Minimum distance between points
                         metric='euclidean',  # Distance metric
                         #random_state=42,
                         )

    X_umap = umap_viz.fit_transform(X_scaled)

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        if mask.sum():
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    if X2 is not None:
        X2_scalled = scaler.transform(X2)
        X2_umap = umap_viz.transform(X2_scalled)
        for i, class_label in enumerate(np.unique(y2)):
            mask = y2 == class_label
            label_name = class_names[i] if class_names else f'Class {class_label}'
            if mask.sum():
                plt.scatter(X2_umap[mask, 0], X2_umap[mask, 1],
                        c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('UMAP Visualization of Classification Data')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if return_umap:
        return X_umap, scaler, umap_viz
    else:
        return X_umap

