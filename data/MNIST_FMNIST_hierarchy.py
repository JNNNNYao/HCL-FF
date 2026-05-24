import json
import argparse
import numpy as np
from typing import List, Dict, Tuple

import torch
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


MNIST_CLASS_NAMES = [str(i) for i in range(10)]
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_dataset(dataset: str, root: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load MNIST or Fashion-MNIST, return (X, y, class_names).
    X shape: (N, 784), y shape: (N,)
    """
    transform = transforms.ToTensor()
    if dataset == "mnist":
        ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        class_names = MNIST_CLASS_NAMES
    elif dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        class_names = FASHION_MNIST_CLASS_NAMES
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'.")

    X = torch.stack([ds[i][0].view(-1) for i in range(len(ds))]).numpy()  # (N, 28*28)
    y = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    return X, y, class_names


def compute_class_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 50,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    PCA -> class prototypes.
    Returns:
      protos: (C, d) class mean features
      class_counts: {class_id: count}
    """
    # Standardize before PCA (good practice for pixel inputs)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_std)

    C = int(y.max()) + 1
    protos = np.zeros((C, n_components), dtype=np.float32)
    class_counts = {}
    for c in range(C):
        idx = (y == c)
        class_counts[c] = int(idx.sum())
        protos[c] = X_pca[idx].mean(axis=0)

    # If using cosine distance later, L2-normalize prototypes
    if metric == "cosine":
        norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
        protos = protos / norms

    return protos, class_counts


def build_linkage(
    protos: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
):
    """Compute scipy linkage from class prototypes."""
    # Ward requires Euclidean distances; others can use cosine etc.
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires Euclidean metric.")
    dist = pdist(protos, metric=metric)  # condensed vector
    Z = linkage(dist, method=method)
    return Z, dist


def export_json(
    Z: np.ndarray,
    class_names: List[str],
    dataset_tag: str,
) -> Dict:
    """
    Export a JSON file.
    - Leaves 0..C-1 are classes
    - Internal nodes get ids hC..h(C+(C-2))
    Produces:
      {
        "directed": true,
        "multigraph": false,
        "graph": {},
        "nodes": [{"id": "...", "label": "...", "is_leaf": bool}, ...],
        "links": [{"source": "...", "target": "..."}, ...]
      }
    """
    C = len(class_names)
    # Prepare nodes
    nodes = []
    id_for_index = {}  # scipy merge index -> node id
    for i in range(C):
        nid = f"{dataset_tag}_leaf_{i}"
        id_for_index[i] = nid
        nodes.append({"id": nid, "label": class_names[i], "is_leaf": True})

    # Build internal nodes by following the linkage merges
    links = []
    next_id_counter = 0
    for row in Z:
        left, right, height, _ = row
        left = int(left)
        right = int(right)
        parent_id = f"{dataset_tag}_int_{next_id_counter}"
        next_id_counter += 1

        # ensure child ids exist
        if left not in id_for_index:
            id_for_index[left] = f"{dataset_tag}_int_auto_{left}"
            nodes.append({"id": id_for_index[left], "label": f"cluster_{left}", "is_leaf": False})
        if right not in id_for_index:
            id_for_index[right] = f"{dataset_tag}_int_auto_{right}"
            nodes.append({"id": id_for_index[right], "label": f"cluster_{right}", "is_leaf": False})

        # create parent node
        nodes.append({"id": parent_id, "label": f"cluster_h{parent_id}", "is_leaf": False})

        # add directed links parent -> children
        links.append({"source": parent_id, "target": id_for_index[left]})
        links.append({"source": parent_id, "target": id_for_index[right]})

        # In scipy indexing, new cluster takes next available integer index
        new_index = max(list(id_for_index.keys()) + [-1]) + 1
        id_for_index[new_index] = parent_id

    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links
    }
    return graph


def main():
    parser = argparse.ArgumentParser(description="Build hierarchy for MNIST/Fashion-MNIST.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--data-root", type=str, default="../datasets")
    parser.add_argument("--n-components", type=int, default=50, help="PCA components")
    parser.add_argument("--method", type=str, default="ward", choices=["ward", "average", "complete", "single"])
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--out-prefix", type=str, default="hierarchy")
    args = parser.parse_args()

    print(f"[1/4] Loading dataset: {args.dataset}")
    X, y, class_names = load_dataset(args.dataset, args.data_root)
    print(f"  -> X: {X.shape}, y: {y.shape}, classes: {class_names}")

    print(f"[2/4] PCA -> class prototypes (n_components={args.n_components}, metric={args.metric})")
    protos, counts = compute_class_prototypes(X, y, n_components=args.n_components, metric=args.metric)
    print(f"  -> prototypes: {protos.shape}")

    print(f"[3/4] Hierarchical clustering (method={args.method}, metric={args.metric})")
    Z, dist = build_linkage(protos, method=args.method, metric=args.metric)
    print("  -> linkage shape:", Z.shape)

    print(f"[4/4] Exporting JSON")
    tag = "mnist" if args.dataset == "mnist" else "fmnist"
    tree = export_json(Z, class_names, dataset_tag=tag)
    tree_path = f"{args.out_prefix}_{args.dataset}.json"
    with open(tree_path, "w") as f:
        json.dump(tree, f, indent=2)
    print(f"  -> wrote tree to {tree_path}")

    print("\nDone. Files generated:")
    print("  -", tree_path)


if __name__ == "__main__":
    main()
