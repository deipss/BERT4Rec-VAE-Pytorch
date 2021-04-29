from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .bert_cnn import BERTCNNModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    BERTCNNModel.code(): BERTCNNModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
