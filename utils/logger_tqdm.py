from tqdm import tqdm as _tqdm


class tqdm(_tqdm):
    logger = None

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)

    def display(self, msg: str=None, pos: int=None) -> None:
        self.logger.info(self.__str__()  if msg is None else msg)
