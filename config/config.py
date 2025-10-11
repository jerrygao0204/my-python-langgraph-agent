import yaml
from typing import Dict, Any

def load_config(file_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    加载并返回 YAML 配置文件内容。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {file_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 错误: 配置文件未找到在路径: {file_path}")
        raise
    except yaml.YAMLError as exc:
        print(f"❌ 错误: 解析 YAML 文件失败: {exc}")
        raise

# 可以在模块级别加载配置，供其他模块直接导入使用
# CONFIG = load_config() 
# 为了测试方便，我们只定义函数，在 main 中调用。
