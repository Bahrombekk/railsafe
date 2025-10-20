"""
Logger konfiguratsiyasi
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = 'logs/railsafe.log', level=logging.INFO):
    """
    Logger yaratish va sozlash
    
    Args:
        name: Logger nomi
        log_file: Log fayl yo'li
        level: Logging darajasi
    
    Returns:
        logging.Logger
    """
    # Logs papkasini yaratish
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logger yaratish
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Agar handler allaqachon qo'shilgan bo'lsa, qaytadan qo'shmaslik
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger