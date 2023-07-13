from argparse import ArgumentParser

DEFAULT_MODEL_NAME = "Chat-Ion"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACC_STEPS = 5
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-1
DEFAULT_CHECKPOINT_INTERVAL = 1_000
DEFAULT_DEVICE = "cuda"
DEFAULT_NUM_GPUS = -1

DEFAULT_DROPUT = 2e-1
DEFAULT_EMBED_DIM = 1020
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_NUM_ENCODER_LAYERS = 12
DEFAULT_NUM_DECODER_LAYERS = 12
DEFAULT_NUM_HEADS = 12
DEFAULT_FEED_FORWARD_DIM = 3072


def get_args() -> dict:
    arg_parser = ArgumentParser()

    train_group = arg_parser.add_argument_group("training")
    train_group.add_argument(
        "-n",
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"model name for checkpointing and saving, defaults to '{DEFAULT_MODEL_NAME}'",
    )
    train_group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"number of epochs to train for, defaults to '{DEFAULT_EPOCHS}'",
    )
    train_group.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batch size, defaults to '{DEFAULT_BATCH_SIZE}'",
    )
    train_group.add_argument(
        "-ga",
        "--grad-acc-steps",
        type=int,
        default=DEFAULT_GRAD_ACC_STEPS,
        help=f"gradient accumulation steps, defaults to '{DEFAULT_GRAD_ACC_STEPS}'",
    )
    train_group.add_argument(
        "-lr",
        "--learning-rate",
        type=int,
        default=DEFAULT_LR,
        help=f"learning rate, defaults to '{DEFAULT_LR}'",
    )
    train_group.add_argument(
        "-wd",
        "--weight-decay",
        type=int,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"weight decay, defaults to '{DEFAULT_WEIGHT_DECAY}'",
    )
    train_group.add_argument(
        "-ci",
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"checkpoint freq in batches, defaults to '{DEFAULT_CHECKPOINT_INTERVAL}'",
    )
    train_group.add_argument(
        "-d",
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"device to train on, defaults to '{DEFAULT_DEVICE}'",
    )
    train_group.add_argument(
        "-ng",
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Num of GPUs to train on in DDP, defaults to all GPU's",
    )

    model_group = arg_parser.add_argument_group("model")
    model_group.add_argument(
        "-dp",
        "--dropout",
        type=float,
        default=DEFAULT_DROPUT,
        help=f"dropout percentage, defaults to '{DEFAULT_DROPUT}'",
    )
    model_group.add_argument(
        "-ed",
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help=f"dimension of embeddings, defaults to '{DEFAULT_EMBED_DIM}'",
    )
    model_group.add_argument(
        "-sl",
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"max seq length, defaults to '{DEFAULT_MAX_SEQ_LEN}'",
    )
    model_group.add_argument(
        "-ne",
        "--num-enc-layers",
        type=int,
        default=DEFAULT_NUM_ENCODER_LAYERS,
        help=f"number of encoder layers, defaults to '{DEFAULT_NUM_ENCODER_LAYERS}'",
    )
    model_group.add_argument(
        "-nd",
        "--num-dec-layers",
        type=int,
        default=DEFAULT_NUM_DECODER_LAYERS,
        help=f"number of decoder layers, defaults to '{DEFAULT_NUM_DECODER_LAYERS}'",
    )
    model_group.add_argument(
        "-nh",
        "--num-heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help=f"number of heads, defaults to '{DEFAULT_NUM_HEADS}'",
    )
    model_group.add_argument(
        "-ff",
        "--feed-forward-dim",
        type=int,
        default=DEFAULT_FEED_FORWARD_DIM,
        help=f"dimension of feed forward layer, defaults to '{DEFAULT_FEED_FORWARD_DIM}'",
    )

    args = arg_parser.parse_args()

    group_args = {}
    for group in arg_parser._action_groups:
        group_args[group.title] = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }

    return group_args
