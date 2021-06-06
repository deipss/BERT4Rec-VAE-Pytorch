import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader, _ = dataloader_factory(args)
    print('template = %s\t model_code = %s\n' % (args.template, args.model_code))
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()

    # test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    # if test_model:

def test50():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader, _ = dataloader_factory(args)
    print('template = %s\t model_code = %s\n' % (args.template, args.model_code))
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.test_top50(_.top50)


if __name__ == '__main__':
    if args.mode == 'train':

        dim_list = [500]
        if args.trainer_code in ['bert', 'ncf']:
            dim_list = [8, 16, 32, 64]
        for i in dim_list:
            args.bert_hidden_units = i
            args.dae_latent_dim = i
            args.vae_latent_dim = i
            args.dim = i
            train()
    if args.mode == 'test50':
        test50()
    else:
        raise ValueError('Invalid mode')
