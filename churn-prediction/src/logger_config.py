"""
Centralized logging configuration for the project.
Always save logs for debugging and auditing.
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json

def setup_logger(name: str, log_dir: Path = None, level=logging.INFO):
    """
    Set up logger with both file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to save logs (creates if doesn't exist)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Format for logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (for production) - rotating logs
    log_file = log_dir / f"{name.split('.')[-1]}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also create a JSON log for structured logging (easier to parse)
    json_formatter = JsonFormatter()
    json_file = log_dir / f"{name.split('.')[-1]}_{datetime.now().strftime('%Y%m%d')}_structured.json"
    json_handler = logging.handlers.RotatingFileHandler(
        json_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    return logger

class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'pathname', 'process', 
                          'processName', 'relativeCreated', 'thread', 
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj)

# Create a specific logger for ML experiments
class MLExperimentLogger:
    """Logger specifically for ML experiments with structured data"""
    
    def __init__(self, experiment_name: str, log_dir: Path = None):
        self.experiment_name = experiment_name
        self.logger = setup_logger(f"ml_experiment.{experiment_name}", log_dir)
        
    def log_data_info(self, df):
        """Log dataset information"""
        self.logger.info("Dataset loaded", extra={
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage().sum() / 1024**2,
            'experiment': self.experiment_name
        })
    
    def log_model_metrics(self, model_name: str, metrics: dict):
        """Log model performance metrics"""
        self.logger.info(f"Model {model_name} trained", extra={
            'model': model_name,
            'metrics': metrics,
            'experiment': self.experiment_name
        })
    
    def log_feature_importance(self, features: dict):
        """Log feature importance"""
        self.logger.info("Feature importance calculated", extra={
            'top_features': dict(list(features.items())[:10]),
            'experiment': self.experiment_name
        })