from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .bert_cnn import BERTCNNModel
from .pop import PopModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    PopModel.code(): PopModel,
    BERTCNNModel.code(): BERTCNNModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
