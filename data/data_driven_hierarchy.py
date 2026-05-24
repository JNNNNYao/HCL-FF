import os
import json
import argparse
import numpy as np
from typing import List, Dict

import torch
import torch.nn.functional as F

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


# --------------------------------------------------
# Class names
# --------------------------------------------------
CIFAR10_CLASS_NAMES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR100_CLASS_NAMES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
TINYIMAGENET200_CLASS_NAMES = [
    "egyptian_cat",
    "reel",
    "volleyball",
    "rocking_chair",
    "lemon",
    "bullfrog",
    "basketball",
    "cliff",
    "espresso",
    "plunger",
    "parking_meter",
    "german_shepherd",
    "dining_table",
    "monarch",
    "brown_bear",
    "school_bus",
    "pizza",
    "guinea_pig",
    "umbrella",
    "organ",
    "oboe",
    "maypole",
    "goldfish",
    "potpie",
    "hourglass",
    "seashore",
    "computer_keyboard",
    "arabian_camel",
    "ice_cream",
    "nail",
    "space_heater",
    "cardigan",
    "baboon",
    "snail",
    "coral_reef",
    "albatross",
    "spider_web",
    "sea_cucumber",
    "backpack",
    "labrador_retriever",
    "pretzel",
    "king_penguin",
    "sulphur_butterfly",
    "tarantula",
    "lesser_panda",
    "pop_bottle",
    "banana",
    "sock",
    "cockroach",
    "projectile",
    "beer_bottle",
    "mantis",
    "freight_car",
    "guacamole",
    "remote_control",
    "european_fire_salamander",
    "lakeside",
    "chimpanzee",
    "pay-phone",
    "fur_coat",
    "alp",
    "lampshade",
    "torch",
    "abacus",
    "moving_van",
    "barrel",
    "tabby",
    "goose",
    "koala",
    "bullet_train",
    "cd_player",
    "teapot",
    "birdhouse",
    "gazelle",
    "academic_gown",
    "tractor",
    "ladybug",
    "miniskirt",
    "golden_retriever",
    "triumphal_arch",
    "cannon",
    "neck_brace",
    "sombrero",
    "gasmask",
    "candle",
    "desk",
    "frying_pan",
    "bee",
    "dam",
    "spiny_lobster",
    "police_van",
    "ipod",
    "punching_bag",
    "beacon",
    "jellyfish",
    "wok",
    "potter's_wheel",
    "sandal",
    "pill_bottle",
    "butcher_shop",
    "slug",
    "hog",
    "cougar",
    "crane",
    "vestment",
    "dragonfly",
    "cash_machine",
    "mushroom",
    "jinrikisha",
    "water_tower",
    "chest",
    "snorkel",
    "sunglasses",
    "fly",
    "limousine",
    "black_stork",
    "dugong",
    "sports_car",
    "water_jug",
    "suspension_bridge",
    "ox",
    "ice_lolly",
    "turnstile",
    "christmas_stocking",
    "broom",
    "scorpion",
    "wooden_spoon",
    "picket_fence",
    "rugby_ball",
    "sewing_machine",
    "steel_arch_bridge",
    "persian_cat",
    "refrigerator",
    "barn",
    "apron",
    "yorkshire_terrier",
    "swimming_trunks",
    "stopwatch",
    "lawn_mower",
    "thatch",
    "fountain",
    "black_widow",
    "bikini",
    "plate",
    "teddy",
    "barbershop",
    "confectionery",
    "beach_wagon",
    "scoreboard",
    "orange",
    "flagpole",
    "american_lobster",
    "trolleybus",
    "drumstick",
    "dumbbell",
    "brass",
    "bow_tie",
    "convertible",
    "bighorn",
    "orangutan",
    "american_alligator",
    "centipede",
    "syringe",
    "go-kart",
    "brain_coral",
    "sea_slug",
    "cliff_dwelling",
    "mashed_potato",
    "viaduct",
    "military_uniform",
    "pomegranate",
    "chain",
    "kimono",
    "comic_book",
    "trilobite",
    "bison",
    "pole",
    "boa_constrictor",
    "poncho",
    "bathtub",
    "grasshopper",
    "walking_stick",
    "chihuahua",
    "tailed_frog",
    "lion",
    "altar",
    "obelisk",
    "beaker",
    "bell_pepper",
    "bannister",
    "bucket",
    "magnetic_compass",
    "meat_loaf",
    "gondola",
    "standard_poodle",
    "acorn",
    "lifeboat",
    "binoculars",
    "cauliflower",
    "african_elephant",
]


# --------------------------------------------------
# Load classifier weights as class prototypes
# --------------------------------------------------
def load_class_prototypes_from_model(
    checkpoint_path: str,
) -> np.ndarray:
    """
    Load pretrained model and extract classifier weights.
    Returns:
      protos: (C, D)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    weight_key = "clf_head.weight"
    if weight_key not in state_dict:
        raise KeyError(f"'{weight_key}' not found in checkpoint. "
                       f"Available keys include: {list(state_dict.keys())[:5]} ...")

    # classifier weights = class prototypes
    protos = F.normalize(state_dict[weight_key], dim=1).cpu().numpy()

    return protos


# --------------------------------------------------
# Hierarchical clustering
# --------------------------------------------------
def build_linkage(
    protos: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
):
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires Euclidean metric.")

    dist = pdist(protos, metric=metric)
    Z = linkage(dist, method=method)
    return Z, dist


# --------------------------------------------------
# Export JSON hierarchy
# --------------------------------------------------
def export_json(
    Z: np.ndarray,
    class_names: List[str],
    dataset_tag: str,
) -> Dict:
    C = len(class_names)

    nodes = []
    links = []
    id_for_index = {}

    # leaves
    for i in range(C):
        nid = f"{dataset_tag}_leaf_{i}"
        id_for_index[i] = nid
        nodes.append({
            "id": nid,
            "label": class_names[i],
            "is_leaf": True
        })

    next_cluster_idx = C
    for k, row in enumerate(Z):
        left, right, height, _ = row
        left, right = int(left), int(right)

        parent_id = f"{dataset_tag}_int_{k}"
        nodes.append({
            "id": parent_id,
            "label": f"cluster_{k}",
            "is_leaf": False
        })

        links.append({"source": parent_id, "target": id_for_index[left]})
        links.append({"source": parent_id, "target": id_for_index[right]})

        id_for_index[next_cluster_idx] = parent_id
        next_cluster_idx += 1

    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build hierarchy from pretrained classifier weights.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet200"])
    parser.add_argument("--method", type=str, default="ward", choices=["ward", "average", "complete", "single"])
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--pretrain_dir", type=str, required=True)
    parser.add_argument("--out-prefix", type=str, default="hierarchy")
    args = parser.parse_args()

    # ----------------------------
    # Dataset config
    # ----------------------------
    if args.dataset == "cifar10":
        class_names = CIFAR10_CLASS_NAMES
        tag = "cifar10"
    elif args.dataset == "cifar100":
        class_names = CIFAR100_CLASS_NAMES
        tag = "cifar100"
    elif args.dataset == "tinyimagenet200":
        class_names = TINYIMAGENET200_CLASS_NAMES
        tag = "tiny200"
    else:
        raise NotImplementedError

    checkpoint_path = os.path.join("../results", args.pretrain_dir, "model.pth")

    print("[1/3] Loading classifier weights")
    protos = load_class_prototypes_from_model(
        checkpoint_path,
    )
    print("  -> prototypes:", protos.shape)

    print("[2/3] Hierarchical clustering")
    Z, _ = build_linkage(protos, method=args.method, metric=args.metric)
    print("  -> linkage:", Z.shape)

    print("[3/3] Exporting JSON")
    tree = export_json(Z, class_names, dataset_tag=tag)
    out_path = f"{args.out_prefix}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2)

    print(f"  -> wrote {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
