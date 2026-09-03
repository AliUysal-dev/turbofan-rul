import logging
import sys


def get_logger(name: str = "turbofan") -> logging.Logger:
    """
    Ö-11 Çözümü: 
    Uvicorn altında mükerrer log oluşmasını engelleyen (propagate=False)
    ve kurumsal standartta formatlanmış merkezi logger üretir.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger