import sys

class Colors:
    """终端颜色常量"""
    # 基础颜色
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 样式
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # 重置
    RESET = '\033[0m'

def colorize(text: str, color_code: str) -> str:
    """辅助函数：将文本包装在颜色代码中并自动重置"""
    return f"{color_code}{text}{Colors.RESET}"

# 预定义的快捷函数
def red(text: str) -> str: return colorize(text, Colors.RED)
def green(text: str) -> str: return colorize(text, Colors.GREEN)
def yellow(text: str) -> str: return colorize(text, Colors.YELLOW)