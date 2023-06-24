from argparse import ArgumentParser

DEFAULT_MODEL_NAME = "Chat-Ion"
DEFAULT_DEVICE = "cuda"

def get_args() -> dict:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"model name to load for evaluation, defaults to '{DEFAULT_MODEL_NAME}'",
    )
    arg_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"device to run evaluation on, defaults to '{DEFAULT_DEVICE}'",
    )

    args = vars(arg_parser.parse_args())

    return args
