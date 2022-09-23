from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer
<<<<<<< HEAD
from .ncf import NCFTrainer
=======
>>>>>>> f66f2534ebfd937778c7174b5f9f216efdebe5de


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
<<<<<<< HEAD
    NCFTrainer.code(): NCFTrainer,
=======
>>>>>>> f66f2534ebfd937778c7174b5f9f216efdebe5de
    VAETrainer.code(): VAETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
