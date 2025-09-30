import logging
import os
import sys

# 日志等级配置
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_logger(name="DHAL", level=None):
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志等级，从环境变量 LOG_LEVEL 读取，默认为 INFO
    
    Returns:
        logger: 配置好的日志器
    """
    if level is None:
        # 从环境变量读取日志等级，默认为 INFO
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = LOG_LEVELS.get(log_level_str, logging.INFO)
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，不重复添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志器
logger = setup_logger()

# 为向后兼容，提供log函数
def log(message, level='INFO'):
    """
    日志函数，向后兼容
    
    Args:
        message: 日志消息
        level: 日志等级
    """
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    logger.log(log_level, message)

# 便捷函数
def debug(message):
    """DEBUG 级别日志"""
    logger.debug(message)

def info(message):
    """INFO 级别日志"""
    logger.info(message)

def warning(message):
    """WARNING 级别日志"""
    logger.warning(message)

def error(message):
    """ERROR 级别日志"""
    logger.error(message)

def critical(message):
    """CRITICAL 级别日志"""
    logger.critical(message)

# 检查是否为 DEBUG 模式
def is_debug_mode():
    """检查是否为 DEBUG 模式"""
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    return log_level_str == 'DEBUG'