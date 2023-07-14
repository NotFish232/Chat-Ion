from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

RUNS_DIR = "tensorboard_runs/"

#writer = SummaryWriter(RUNS_DIR)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', RUNS_DIR])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")