import os
from pathlib import Path

def add_path_headers_to_project(root_dir: str):
    root_path = Path(root_dir)
    
    # 遍历项目下所有的 .py 文件
    for file_path in root_path.rglob("*.py"):
        
        # 对 __init__.py 排除
        if file_path.name in ["add_headers.py", "__init__.py"] or "venv" in file_path.parts or ".git" in file_path.parts:
            continue
            
        # 获取相对路径，并强制转换为 POSIX 风格 (正斜杠 /)
        relative_path = file_path.relative_to(root_path).as_posix()
        header = f'"""{relative_path}"""\n\n'
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 检查是否已经包含了这段头注释，防止重复添加
        if not content.startswith(f'"""{relative_path}"""'):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(header + content)
                print(f"已处理: {relative_path}")

if __name__ == "__main__":
    # '.' 代表当前项目根目录
    add_path_headers_to_project(".")
    print("全部搞定！")