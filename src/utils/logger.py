"""
src/utils/logger.py
Merkezi ve standartlaştırılmış konsol/dosya günlükleme yapılandırması.
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str = "turbofan_rul", log_file: Path | None = None) -> logging.Logger:
    """Endüstriyel standartta formatlanmış bir logger döndürür."""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger