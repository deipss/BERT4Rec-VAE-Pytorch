from .ml_1m import ML1MDataset
from .ml_10m import ML10MDataset
from .ml_100k import ML100KDataset
from .ml_20m import ML20MDataset
from .behavior import BehaviorDataset
from .app import AppDataset
from .fashion import FashionDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML100KDataset.code(): ML100KDataset,
    ML20MDataset.code(): ML20MDataset,
    ML10MDataset.code(): ML10MDataset,
    BehaviorDataset.code(): BehaviorDataset,
    AppDataset.code(): AppDataset,
    FashionDataset.code(): FashionDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
