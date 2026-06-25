import logging

class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    BLUE = "\033[94m"    # Used for DEBUG/VERBOSE
    GREEN = "\033[92m"   # Used for INFO
    YELLOW = "\033[93m"  # Used for WARNING
    RED = "\033[91m"     # Used for ERROR and CRITICAL
    RESET = "\033[0m"    # Reset color to default

    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        # Apply the color to the entire log message line
        format_str = f"{log_color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{self.RESET}"
        formatter = logging.Formatter(format_str)
        return formatter.format(record)

def setup_logger(name):
    """Creates a logger with colored console output."""
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent logs from propagating up to the root logger which might double-print
        logger.propagate = False 
    return logger
