#!/bin/bash

echo "启动 LongCat-Image Streamlit 网页界面..."
echo "=================================================="
echo ""
echo "请确保您已经:"
echo "1. 安装了所有依赖: pip install -r requirements.txt"
echo "2. 将模型下载到 ./weights/ 目录"
echo ""
echo "应用将在您的默认浏览器中打开 http://localhost:8501"
echo ""

streamlit run app.py
