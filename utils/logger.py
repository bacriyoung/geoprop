import logging
import os
import sys

def setup_logger(output_dir, name="geoprop", log_filename="pipeline.log"):
    """
    Args:
        output_dir: 
        name: Logger
        log_filename:(e.g. pipeline_20251230_120000.log)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    log_file = os.path.join(output_dir, log_filename)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger