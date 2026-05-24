from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

def json_to_hierarchy(
    graph_json: Dict[str, Any],
    class_order: List[str],
    max_levels: Optional[int] = None,
    label_aliases: Optional[Dict[str, str]] = None,
) -> Dict[int, List[List[int]]]:
    """
    Convert JSON -> {level: [[class_idx,...], ...]} with guarantees:
      • Works when some nodes lack 'label' (e.g., 'f0000...' internal nodes).
      • Accepts label_aliases to normalize leaf labels to dataset names.
      • At EVERY level, all leaves (0..C-1) appear exactly once (partition).
    """

    label_aliases = label_aliases or {}

    # --- Parse nodes/edges (handle unlabeled nodes) ---
    nodes = graph_json.get("nodes", [])
    links = graph_json.get("links", [])
    # Only include labeled nodes in label maps
    id2label_raw = {n["id"]: n["label"] for n in nodes if "label" in n}
    # Normalize leaf labels via aliases so they match class_order names
    id2label_norm = {nid: label_aliases.get(lbl, lbl) for nid, lbl in id2label_raw.items()}
    # Build reverse map using normalized labels (last one wins if duplicates)
    label_norm2id = {lbl: nid for nid, lbl in id2label_norm.items()}

    children, parents = defaultdict(list), defaultdict(list)
    for e in links:
        s, t = e["source"], e["target"]
        children[s].append(t)
        parents[t].append(s)

    # --- Dataset leaves & indices (dataset names are the target names) ---
    C = len(class_order)
    dataset_label2idx = {lbl: i for i, lbl in enumerate(class_order)}
    # Leaf nodes we care about are those whose *normalized* label exists in class_order
    leaf_ids = {label_norm2id[lbl] for lbl in class_order if lbl in label_norm2id}

    if not leaf_ids:
        raise ValueError("No leaf labels match your dataset class names. "
                         "Provide label_aliases to normalize leaf labels.")

    # --- Find root ---
    cand_roots = [nid for nid in set(children) | set(parents) if len(parents[nid]) == 0]
    if cand_roots:
        def count_dataset_leaves(start_id: str) -> int:
            q, seen, cnt = deque([start_id]), {start_id}, 0
            while q:
                u = q.popleft()
                if u in leaf_ids: cnt += 1
                for v in children.get(u, []):
                    if v not in seen:
                        seen.add(v); q.append(v)
            return cnt
        root_id = max(cand_roots, key=count_dataset_leaves)
    else:
        raise ValueError("No root found (every node has a parent?).")

    # --- Descendant leaves -> dataset class indices ---
    def descendant_leaf_idxs(node_id: str) -> List[int]:
        q, seen, leaves = deque([node_id]), {node_id}, set()
        while q:
            u = q.popleft()
            if u in leaf_ids:
                leaves.add(u); continue
            for v in children.get(u, []):
                if v not in seen:
                    seen.add(v); q.append(v)
        # map leaf ids -> normalized labels -> dataset indices
        idxs = []
        for lid in leaves:
            lbl_norm = id2label_norm.get(lid)
            if lbl_norm is None:  # safety: unlabeled leaf (shouldn't happen), skip
                continue
            idx = dataset_label2idx.get(lbl_norm)
            if idx is not None:
                idxs.append(idx)
        return sorted(set(idxs))

    # --- BFS to collect nodes by level (depth from root) ---
    level_nodes: Dict[int, List[str]] = defaultdict(list)
    q, seen = deque([(root_id, 0)]), {root_id}
    level_nodes[0].append(root_id)
    max_depth = 0
    while q:
        u, d = q.popleft()
        for v in children.get(u, []):
            if v not in seen:
                seen.add(v)
                level_nodes[d + 1].append(v)
                q.append((v, d + 1))
                max_depth = max(max_depth, d + 1)

    # --- Build groups per level, enforcing "all leaves at all levels" ---
    deepest = max_depth if max_levels is None else min(max_depth, max_levels)
    all_idxs = set(range(C))
    hierarchy: Dict[int, List[List[int]]] = {}

    for L in range(1, deepest + 1):
        groups = []
        covered = set()
        for nid in level_nodes.get(L, []):
            g = descendant_leaf_idxs(nid)
            if g:
                groups.append(g)
                covered.update(g)
        # Add singletons for any missing leaves (branches that ended earlier)
        missing = sorted(all_idxs - covered)
        groups.extend([[i] for i in missing])
        if not groups:
            break
        hierarchy[L] = groups  # keep BFS order, singletons appended

    # If the tree is shallow and produced no levels, emit an identity level
    if not hierarchy:
        hierarchy[1] = [[i] for i in range(C)]

    # --- Helpful warning for unmatched labels ---
    leaf_labels = {id2label_norm[lid] for lid in leaf_ids if lid in id2label_norm}
    missing_in_graph = [lbl for lbl in class_order if lbl not in leaf_labels]
    if missing_in_graph:
        print(f"[json_to_hierarchy] Warning: {len(missing_in_graph)} dataset classes "
              f"not found in JSON (after aliasing), e.g.: {missing_in_graph[:10]} ...")

    return hierarchy


# ---------------------- Example: your JSON -> hierarchy ----------------------

if __name__ == "__main__":
    import json

    # Load your MNIST exported JSON file
    with open("./hierarchy_mnist.json", "r") as f:
        json_data = json.load(f)

    # MNIST class order → indices
    mnist_order = [str(i) for i in range(10)]

    hier = json_to_hierarchy(json_data, mnist_order)
    print("hierarchy = {")
    for lvl in sorted(hier.keys()):
        print(f"    {lvl}: {hier[lvl]} / {len(hier[lvl])},")
    print("}")

    # Load your FMNIST exported JSON file
    with open("./hierarchy_fashion_mnist.json", "r") as f:
        json_data = json.load(f)
    
    # FMNIST class order → indices
    fmnist_order = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    hier = json_to_hierarchy(json_data, fmnist_order)
    print("hierarchy = {")
    for lvl in sorted(hier.keys()):
        print(f"    {lvl}: {hier[lvl]} / {len(hier[lvl])},")
    print("}")

    # Load your CIFAR-10 exported JSON file
    with open("./hierarchy_cifar10.json", "r") as f:
        json_data = json.load(f)

    # CIFAR-10 class order → indices
    cifar10_order = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    hier = json_to_hierarchy(json_data, cifar10_order)
    print("hierarchy = {")
    for lvl in sorted(hier.keys()):
        print(f"    {lvl}: {hier[lvl]} / {len(hier[lvl])},")
    print("}")

    # Load your CIFAR-100 exported JSON file
    with open("./hierarchy_cifar100.json", "r") as f:
        json_data = json.load(f)

    # CIFAR-100 class order → indices
    cifar100_order = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    hier = json_to_hierarchy(json_data, cifar100_order)
    print("hierarchy = {")
    for lvl in sorted(hier.keys()):
        print(f"    {lvl}: {hier[lvl]} / {len(hier[lvl])},")
    print("}")

    with open("./hierarchy_tinyimagenet200.json", "r") as f:
        json_data = json.load(f)
    
    tiny_imagenet_200_classes = [
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
    
    aliases = {
        
    }
    
    hier = json_to_hierarchy(json_data, tiny_imagenet_200_classes, label_aliases=aliases)
    print("hierarchy = {")
    for lvl in sorted(hier.keys()):
        print(f"    {lvl}: {len(hier[lvl])},")
    print("}")