from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .bert_bilstm import BERTBILSTMModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    BERTBILSTMModel.code(): BERTBILSTMModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
