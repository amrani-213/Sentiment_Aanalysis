import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name='sentiment_analysis', log_dir='logs', level=logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'{name}_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_metrics(logger, metrics, prefix='', step=None):
    log_str = f"{prefix} " if prefix else ""
    
    if step is not None:
        log_str += f"Step {step} - "
    
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{key}: {value:.4f}")
        else:
            metric_strs.append(f"{key}: {value}")
    
    log_str += ", ".join(metric_strs)
    logger.info(log_str)


def log_training_step(logger, epoch, total_epochs, batch, total_batches, 
                      loss, accuracy, lr=None):
    log_str = f"Epoch [{epoch}/{total_epochs}] Batch [{batch}/{total_batches}] "
    log_str += f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    
    if lr is not None:
        log_str += f", LR: {lr:.6f}"
    
    logger.info(log_str)