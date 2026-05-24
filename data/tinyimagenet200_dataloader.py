import os
import copy
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .build_hierarchy import json_to_hierarchy


class InverseNormalize:
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[None, :, None, None]
        self.std = torch.Tensor(std)[None, :, None, None]

    def __call__(self, sample):
        return (sample * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class TinyImagenet200(Dataset):
    """Tiny imagenet dataloader"""

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    dataset = None

    def __init__(self, root="./data", *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            self.download(root=root)
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )

    @staticmethod
    def transform_val():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )

    @staticmethod
    def transform_val_inverse():
        return InverseNormalize(
            [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
        )

    def download(self, root="./"):
        """Download and unzip Imagenet200 files in the `root` directory."""
        dir = os.path.join(root, "tiny-imagenet-200")
        dir_train = os.path.join(dir, "train")
        if os.path.exists(dir) and os.path.exists(dir_train):
            print("==> Already downloaded.")
            return

        path = Path(os.path.join(root, "tiny-imagenet-200.zip"))
        if not os.path.exists(path):
            os.makedirs(path.parent, exist_ok=True)

            print("==> Downloading TinyImagenet200...")
            with urllib.request.urlopen(self.url) as response, open(
                str(path), "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

        print("==> Extracting TinyImagenet200...")
        with zipfile.ZipFile(str(path)) as zf:
            zf.extractall(root)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/train"), *args, **kwargs)


class _TinyImagenet200Val(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/val"), *args, **kwargs)

        self.path_to_class = {}
        with open(os.path.join(self.root, "val_annotations.txt")) as f:
            for line in f.readlines():
                parts = line.split()
                path = os.path.join(self.root, "images", parts[0])
                self.path_to_class[path] = parts[1]

        self.classes = list(sorted(set(self.path_to_class.values())))
        self.class_to_idx = {label: self.classes.index(label) for label in self.classes}

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        path, _ = self.samples[i]
        label = self.path_to_class[path]
        target = self.class_to_idx[label]
        return sample, target

    def __len__(self):
        return super().__len__()


def get_tinyimagenet200_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    train_dataset = TinyImagenet200(root=root, train=True, download=True, transform=TinyImagenet200.transform_train(input_size=64))
    test_dataset = TinyImagenet200(root=root, train=False, download=True, transform=TinyImagenet200.transform_val())

    with open("./data/hierarchy_tinyimagenet200.json", "r") as f:
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
    hierarchy = json_to_hierarchy(
        graph_json=json_data,
        class_order=tiny_imagenet_200_classes,
        label_aliases=aliases
    )

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [90000, 10000])
    
    valid_dataset = copy.deepcopy(valid_dataset)
    valid_dataset.dataset.transform = TinyImagenet200.transform_val()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader, hierarchy