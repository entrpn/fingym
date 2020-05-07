import logging

def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger