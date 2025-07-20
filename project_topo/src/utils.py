import os
import torch
import json

def load_config():
    """Tải cấu hình từ file config.json hoặc trả về mặc định."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    default_config = {"project_root": os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return {**default_config, **json.load(f)}
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy config.json, sử dụng giá trị mặc định: {default_config}")
        return default_config

def save_model(model, path):
    """Lưu trạng thái mô hình."""
    torch.save(model.state_dict(), path)

def load_model(path):
    """Tải trạng thái mô hình."""
    return torch.load(path)

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại."""
    os.makedirs(directory, exist_ok=True)