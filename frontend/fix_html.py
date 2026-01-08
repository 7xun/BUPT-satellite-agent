# -*- coding: utf-8 -*-
"""
HTML 资源路径修复脚本
"""
import base64
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(CURRENT_DIR, "satellite.html")
IMAGE_PATH = os.path.join(CURRENT_DIR, "images", "p1.PNG")

def main():
    if not os.path.exists(HTML_PATH):
        print(f"文件不存在: {HTML_PATH}")
        return

    # 读取 HTML
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 读取图片并转 Base64
    if os.path.exists(IMAGE_PATH):
        with open(IMAGE_PATH, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
            b64_src = f"data:image/png;base64,{b64_data}"
            
        # 替换图片路径
        content = content.replace('src="images/p1.PNG"', f'src="{b64_src}"')
        content = content.replace('src="./images/p1.PNG"', f'src="{b64_src}"')

    # 替换 ECharts CDN
    content = content.replace(
        '<script src="./js/echarts.min.js"></script>',
        '<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>'
    )

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML 修复完成")

if __name__ == "__main__":
    main()
