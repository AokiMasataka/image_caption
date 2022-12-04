import logging


class AvgManager:
    def __init__(self):
        self.value = 0.0
        self.n = 0
    
    def __len__(self):
        assert isinstance(self.n, int)
        return self.n
    
    def __call__(self):
        return self.value / self.n
    
    def update(self, value):
        self.value += value
        self.n += 1


def set_logger(log_file):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)