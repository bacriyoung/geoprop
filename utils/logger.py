import logging
import os
import sys

def setup_logger(output_dir, name="geoprop"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    # 将日志文件放在 outputs/s3dis/logs/ 下
    # 传入的 output_dir 是 logs 文件夹
    log_file = os.path.join(output_dir, 'pipeline.log')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
