import os
import tensorboard
import time

RUNS_DIR = "tensorboard_runs/"


def main() -> None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', RUNS_DIR])
    url = tb.launch()
    print(f"Running at \"{url}\"")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()