from .ml_1m import ML1MDataset
from .ml_100k import ML100KDataset
from .ml_20m import ML20MDataset
from .db import DbDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML100KDataset.code(): ML100KDataset,
    ML20MDataset.code(): ML20MDataset,
    DbDataset.code(): DbDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
