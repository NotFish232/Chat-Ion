from argparse import ArgumentParser, Namespace


DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_DEVICE = "cuda"
DEFAULT_CHECKPOINT = 2
DEFAULT_EMBED_DIM = 512
DEFAULT_SRC_SEQ_LEN = 15
DEFAULT_TGT_SEQ_LEN = 15
DEFAULT_NUM_ENCODER_LAYERS = 6
DEFAULT_NUM_DECODER_LAYERS = 6
DEFAULT_MODEL_NAME = "model"


def get_args() -> Namespace:
    args = ArgumentParser()

    args.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"number of epochs to train for, defaults to '{DEFAULT_EPOCHS}'",
    )
    args.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batch size, defaults to '{DEFAULT_BATCH_SIZE}'",
    )
    args.add_argument(
        "--learning-rate",
        type=int,
        default=DEFAULT_LR,
        help=f"learning rate, defaults to '{DEFAULT_LR}'"
    )
    args.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"device to train on, defaults to '{DEFAULT_DEVICE}'",
    )
    args.add_argument(
        "--checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINT,
        help=f"checkpoint freq in epochs, defaults to '{DEFAULT_CHECKPOINT}'",
    )
    args.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help=f"dimension of embeddings, defaults to '{DEFAULT_EMBED_DIM}'",
    )
    args.add_argument(
        "--src-seq-len",
        type=int,
        default=DEFAULT_SRC_SEQ_LEN,
        help=f"max seq len of input, defaults to '{DEFAULT_SRC_SEQ_LEN}'",
    )
    args.add_argument(
        "--tgt-seq-len",
        type=int,
        default=DEFAULT_TGT_SEQ_LEN,
        help=f"max seq len of output, defaults to '{DEFAULT_TGT_SEQ_LEN}'",
    )
    args.add_argument(
        "--num-enc-layers",
        type=int,
        default=DEFAULT_NUM_ENCODER_LAYERS,
        help=f"number of encoder layers, defaults to '{DEFAULT_NUM_ENCODER_LAYERS}'",
    )
    args.add_argument(
        "--num-dec-layers",
        type=int,
        default=DEFAULT_NUM_DECODER_LAYERS,
        help=f"number of decoder layers, defaults to '{DEFAULT_NUM_DECODER_LAYERS}'",
    )
    args.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"model name for checkpointing and saving, defaults to '{DEFAULT_MODEL_NAME}'",
    )

    return args.parse_args()
