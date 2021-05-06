import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def init():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader ,dataloader= dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    best_model = torch.load(os.path.join('./best/best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    global E
    global mid2idx
    global midx2id
    mid2idx = dataloader.smap
    midx2id = dataloader.sidmap
    E = model.bert.embedding.token.weight.data
    print(E.shape)


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader,dataloader = dataloader_factory(args)
    print('template = %s\t model_code = %s\n' % (args.template, args.model_code))
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
    best_model = torch.load(os.path.join('./best/best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    global E
    global mid2idx
    global midx2id
    mid2idx = dataloader.smap
    midx2id = dataloader.sidmap
    E = model.bert.embedding.token.weight.data


E = {}
mid2idx = {}
midx2id = {}
if __name__ == '__main__':
    pass
    if args.mode == 'train':
        i = 64
        args.bert_hidden_units = i
        args.dae_latent_dim = i
        args.vae_latent_dim = i
        args.dim = i
        train()
        print(E.shape)
        print(len(mid2idx))
        print(len(midx2id))
    else:
        raise ValueError('Invalid mode')
