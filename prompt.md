帮我编写python代码，实现一个用户友好的 Streamlit Web 界面，用于 LongCat-Image

LongCat-Image参考文件：
- README.md

===============


> 请根据prompt.md中的指令执行任务

● I'll read the prompt.md file to understand what task needs to be executed.

● Read(prompt.md)
  ⎿  Read 5 lines

● Now let me read the README.md to understand what LongCat-Image is and what functionality needs to be implemented.

● Read(README.md)
  ⎿  Read 420 lines

● Now I understand the task. I need to create a user-friendly Streamlit web interface for LongCat-Image that
  supports both Text-to-Image generation and Image Editing functionality. Let me create a comprehensive todo list
  and then implement the Streamlit app.

● Search(pattern: "**/*.py")
  ⎿  Found 27 files (ctrl+o to expand)

● Bash(ls -la)
  ⎿  total 108
     drwxr-xr-x  8 ubuntu ubuntu  4096 Dec  9 21:44 .
     drwxr-xr-x 10 ubuntu ubuntu  4096 Dec  9 20:51 ..
     … +13 lines (ctrl+o to expand)

● Let me check the existing inference scripts to understand how the models are used:

● Read(scripts/inference_t2i.py)
  ⎿  Read 45 lines

● Read(scripts/inference_edit.py)
  ⎿  Read 38 lines

● Read(requirements.txt)
  ⎿  Read 48 lines

● Now I have a good understanding of the project structure. Let me update the todo list and create a comprehensive
  Streamlit web interface:

● Now let me create a comprehensive Streamlit web interface:

● Write(app.py)
  ⎿  Wrote 314 lines to app.py
     import streamlit as st
     import torch
     from PIL import Image
     import io
     from transformers import AutoProcessor
     from longcat_image.models import LongCatImageTransformer2DModel
     from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline
     st.set_page_config(
         page_title="LongCat-Image Web Interface",
     … +304 lines (ctrl+o to expand)

● Now let me update the todo list and create a README for running the Streamlit app:

● Write(STREAMLIT_README.md)
  ⎿  Wrote 160 lines to STREAMLIT_README.md
     # LongCat-Image Streamlit Web Interface
     A user-friendly web interface for LongCat-Image built with Streamlit, supporting both Text-to-Image generation
     and Image Editing.
     ## Features
     - **Text-to-Image Generation**: Generate high-quality images from text prompts in English or Chinese
     - **Image Editing**: Edit existing images with natural language instructions
     - **Bilingual Support**: Full support for both Chinese and English text rendering
     - **User-Friendly Interface**: Intuitive web interface with real-time parameter adjustment

     … +150 lines (ctrl+o to expand)

● Now let me update the requirements.txt to include streamlit:

● Read(requirements.txt)
  ⎿  Read 48 lines

● Update(requirements.txt)
  ⎿  Updated requirements.txt with 2 additions and 1 removal
       45    urllib3==2.5.0
       46    zipp==3.23.0
       47    openai==2.8.1
       48 -  peft==0.18.0
       48   \ No newline at end of file
       49 +  peft==0.18.0
       50 +  streamlit==1.40.2
       51   \ No newline at end of file

● Now let me complete the todos and create a quick start script:

● Write(run_app.sh)
  ⎿  Wrote 14 lines to run_app.sh
     #!/bin/bash
     echo "Starting LongCat-Image Streamlit Web Interface..."
     echo "=================================================="
     echo ""
     echo "Make sure you have:"
     echo "1. Installed all dependencies: pip install -r requirements.txt"
     echo "2. Downloaded the models to ./weights/ directory"
     echo ""
     echo "The app will open in your default browser at http://localhost:8501"
     … +4 lines (ctrl+o to expand)

● Bash(chmod +x /home/ubuntu/github/LongCat-Image/run_app.sh)
  ⎿  Tool use rejected with user message: 界面需要是中文的

● 好的,我需要将界面改为中文。让我更新Streamlit应用:

● Write(app.py)
  ⎿  Updated app.py with 71 additions and 71 removals
        7    from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline
        8
        9    st.set_page_config(
       10 -      page_title="LongCat-Image Web Interface",
       10 +      page_title="LongCat-Image 网页界面",
       11        page_icon="🐱",
       12        layout="wide"
       13    )
     ...
        65        return pipe
        66
        67    def main():
        68 -      st.title("🐱 LongCat-Image Web Interface")
        69 -      st.markdown("### Bilingual Text-to-Image Generation and Image Editing")
        68 +      st.title("🐱 LongCat-Image 网页界面")
        69 +      st.markdown("### 中英双语文生图与图像编辑")
        70
        71 -      st.sidebar.header("Model Configuration")
        71 +      st.sidebar.header("模型配置")
        72
        73        t2i_checkpoint = st.sidebar.text_input(
        74 -          "Text-to-Image Model Path",
        74 +          "文生图模型路径",
        75            value="./weights/LongCat-Image",
        76 -          help="Path to LongCat-Image checkpoint directory"
        76 +          help="LongCat-Image 模型检查点目录路径"
        77        )
        78
        79        edit_checkpoint = st.sidebar.text_input(
        80 -          "Image Edit Model Path",
        80 +          "图像编辑模型路径",
        81            value="./weights/LongCat-Image-Edit",
        82 -          help="Path to LongCat-Image-Edit checkpoint directory"
        82 +          help="LongCat-Image-Edit 模型检查点目录路径"
        83        )
        84
        85        use_cpu_offload = st.sidebar.checkbox(
        86 -          "Enable CPU Offload",
        86 +          "启用 CPU 卸载",
        87            value=True,
        88 -          help="Enable to save VRAM (slower but prevents OOM). Disable for faster inference on high
           - VRAM devices."
        88 +          help="启用可节省显存(速度较慢但避免显存溢出)。在高显存设备上禁用可获得更快的推理速度。"
        89        )
        90
        91 -      tab1, tab2, tab3 = st.tabs(["📝 Text-to-Image", "✏️ Image Editing", "ℹ️ About"])
        91 +      tab1, tab2, tab3 = st.tabs(["📝 文生图", "✏️ 图像编辑", "ℹ️ 关于"])
        92
        93        with tab1:
        94 -          st.header("Text-to-Image Generation")
        95 -          st.info("⚠️ **Special Handling for Text Rendering**: When generating text in images,
           - enclose the target text within quotes (\"\") for better quality.")
        94 +          st.header("文生图生成")
        95 +          st.info("⚠️ **文字渲染特殊处理**:
           + 当生成包含文字的图像时,请将目标文字用引号(\"\")括起来以获得更好的质量。")
        96
        97            col1, col2 = st.columns([1, 1])
        98
        99            with col1:
       100                prompt = st.text_area(
       101 -                  "Prompt",
       101 +                  "提示词",
       102                    value='一个年轻的亚裔女性,身穿黄色针织衫,搭配白色项链。她的双手放在膝盖上,表情恬静
             。背景是一堵粗糙的砖墙,午后的阳光温暖地洒在她身上,营造出一种宁静而温馨的氛围。',
       103                    height=150,
       104 -                  help="Enter your image generation prompt in English or Chinese"
       104 +                  help="输入中文或英文的图像生成提示词"
       105                )
       106
       107                negative_prompt = st.text_area(
       108 -                  "Negative Prompt (Optional)",
       108 +                  "负面提示词(可选)",
       109                    value='',
       110                    height=80,
       111 -                  help="Describe what you don't want in the image"
       111 +                  help="描述您不希望在图像中出现的内容"
       112                )
       113
       114                col1_1, col1_2 = st.columns(2)
       115                with col1_1:
       116 -                  width = st.slider("Width", min_value=512, max_value=2048, value=1344, step=64)
       117 -                  guidance_scale = st.slider("Guidance Scale
           -  ", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
       118 -                  enable_cfg_renorm = st.checkbox("Enable CFG Renorm", value=True)
       116 +                  width = st.slider("宽度", min_value=512, max_value=2048, value=1344, step=64)
       117 +                  guidance_scale = st.slider("引导强度
           +  ", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
       118 +                  enable_cfg_renorm = st.checkbox("启用 CFG 重归一化", value=True)
       119
       120                with col1_2:
       121 -                  height = st.slider("Height", min_value=512, max_value=2048, value=768, step=64)
       122 -                  num_inference_steps = st.slider("Inference Steps
           -  ", min_value=10, max_value=100, value=50, step=5)
       123 -                  enable_prompt_rewrite = st.checkbox("Enable Prompt Rewrite", value=True, help="
           -  Reuse text encoder as built-in prompt rewriter")
       121 +                  height = st.slider("高度", min_value=512, max_value=2048, value=768, step=64)
       122 +                  num_inference_steps = st.slider("推理步数
           +  ", min_value=10, max_value=100, value=50, step=5)
       123 +                  enable_prompt_rewrite = st.checkbox("启用提示词重写", value=True,
     help="使用内置的文本编码器作为提示词改写器
           +  ")
       124
       125 -              num_images = st.number_input("Number of Images", min_value=1, max_value=4, value=1)
       126 -              seed = st.number_input("Seed", min_value=-1, max_value=999999, value=43, help="Use -1
           -  for random seed")
       125 +              num_images = st.number_input("图像数量", min_value=1, max_value=4, value=1)
       126 +              seed = st.number_input("随机种子", min_value=-1, max_value=999999, value=43, help="使用
     -1
           +  表示随机种子")
       127
       128 -              generate_button = st.button("🎨 Generate Image
           -  ", type="primary", use_container_width=True)
       128 +              generate_button = st.button("🎨 生成图像", type="primary", use_container_width=True)
       129
       130            with col2:
       131                if generate_button:
       132                    try:
       133 -                      with st.spinner("Loading model..."):
       133 +                      with st.spinner("正在加载模型..."):
       134                            pipe = load_t2i_model(t2i_checkpoint, use_cpu_offload)
       135
       136 -                      with st.spinner("Generating image... This may take a while."):
       136 +                      with st.spinner("正在生成图像... 这可能需要一段时间。"):
       137                            generator = torch.Generator("cpu").manual_seed(seed) if seed >= 0 else None
       138
       139                            result = pipe(
     ...
       151
       152                            images = result.images
       153
       154 -                          st.success(f"✅ Generated {len(images)} image(s) successfully!")
       154 +                          st.success(f"✅ 成功生成 {len(images)} 张图像!")
       155
       156                            for idx, image in enumerate(images):
       157 -                              st.image(image, caption=f"Generated Image
           -  {idx + 1}", use_container_width=True)
       157 +                              st.image(image, caption=f"生成图像 {idx + 1}", use_container_width=True)
       158
       159                                buf = io.BytesIO()
       160                                image.save(buf, format='PNG')
       161                                buf.seek(0)
       162
       163                                st.download_button(
       164 -                                  label=f"⬇️ Download Image {idx + 1}",
       164 +                                  label=f"⬇️ 下载图像 {idx + 1}",
       165                                    data=buf,
       166                                    file_name=f"longcat_t2i_{idx + 1}.png",
       167                                    mime="image/png",
     ...
       169                                )
       170
       171                    except Exception as e:
       172 -                      st.error(f"❌ Error generating image: {str(e)}")
       172 +                      st.error(f"❌ 生成图像时出错: {str(e)}")
       173                        st.exception(e)
       174                else:
       175 -                  st.info("👈 Configure parameters and click 'Generate Image' to start")
       175 +                  st.info("👈 配置参数后点击'生成图像'开始")
       176
       177        with tab2:
       178 -          st.header("Image Editing")
       179 -          st.info("⚠️ **Special Handling for Text Rendering**: When editing images with text, enclose
           -  the target text within quotes (\"\") for better quality.")
       178 +          st.header("图像编辑")
       179 +          st.info("⚠️ **文字渲染特殊处理**:
           + 当编辑包含文字的图像时,请将目标文字用引号(\"\")括起来以获得更好的质量。")
       180
       181            col1, col2 = st.columns([1, 1])
       182
       183            with col1:
       184                uploaded_file = st.file_uploader(
       185 -                  "Upload Image",
       185 +                  "上传图像",
       186                    type=['png', 'jpg', 'jpeg'],
       187 -                  help="Upload an image to edit"
       187 +                  help="上传要编辑的图像"
       188                )
       189
       190                if uploaded_file is not None:
       191                    input_image = Image.open(uploaded_file).convert('RGB')
       192 -                  st.image(input_image, caption="Input Image", use_container_width=True)
       192 +                  st.image(input_image, caption="输入图像", use_container_width=True)
       193
       194                edit_prompt = st.text_area(
       195 -                  "Edit Instruction",
       195 +                  "编辑指令",
       196                    value='将猫变成狗',
       197                    height=100,
       198 -                  help="Describe how you want to edit the image"
       198 +                  help="描述您想如何编辑图像"
       199                )
       200
       201                edit_negative_prompt = st.text_area(
       202 -                  "Negative Prompt (Optional)",
       202 +                  "负面提示词(可选)",
       203                    value='',
       204                    height=80,
       205 -                  help="Describe what you don't want in the edited image"
       205 +                  help="描述您不希望在编辑后的图像中出现的内容"
       206                )
       207
       208                col2_1, col2_2 = st.columns(2)
       209                with col2_1:
       210 -                  edit_guidance_scale = st.slider("Guidance Scale
           -  ", min_value=1.0, max_value=10.0, value=4.5, step=0.1, key="edit_guidance")
       211 -                  edit_num_images = st.number_input("Number of Images
           -  ", min_value=1, max_value=4, value=1, key="edit_num_images")
       210 +                  edit_guidance_scale = st.slider("引导强度
           +  ", min_value=1.0, max_value=10.0, value=4.5, step=0.1, key="edit_guidance")
       211 +                  edit_num_images = st.number_input("图像数量
           +  ", min_value=1, max_value=4, value=1, key="edit_num_images")
       212
       213                with col2_2:
       214 -                  edit_num_inference_steps = st.slider("Inference Steps
           -  ", min_value=10, max_value=100, value=50, step=5, key="edit_steps")
       215 -                  edit_seed = st.number_input("Seed
           -  ", min_value=-1, max_value=999999, value=43, help="Use -1 for random seed", key="edit_seed")
       214 +                  edit_num_inference_steps = st.slider("推理步数
           +  ", min_value=10, max_value=100, value=50, step=5, key="edit_steps")
       215 +                  edit_seed = st.number_input("随机种子
           +  ", min_value=-1, max_value=999999, value=43, help="使用 -1 表示随机种子", key="edit_seed")
       216
       217 -              edit_button = st.button("✏️ Edit Image
           -  ", type="primary", use_container_width=True, disabled=(uploaded_file is None))
       217 +              edit_button = st.button("✏️ 编辑图像
           +  ", type="primary", use_container_width=True, disabled=(uploaded_file is None))
       218
       219            with col2:
       220                if edit_button and uploaded_file is not None:
       221                    try:
       222 -                      with st.spinner("Loading model..."):
       222 +                      with st.spinner("正在加载模型..."):
       223                            edit_pipe = load_edit_model(edit_checkpoint, use_cpu_offload)
       224
       225 -                      with st.spinner("Editing image... This may take a while."):
       225 +                      with st.spinner("正在编辑图像... 这可能需要一段时间。"):
       226                            edit_generator = torch.Generator("cpu").manual_seed(edit_seed) if edit_seed
              >= 0 else None
       227
       228                            result = edit_pipe(
     ...
       237
       238                            images = result.images
       239
       240 -                          st.success(f"✅ Edited {len(images)} image(s) successfully!")
       240 +                          st.success(f"✅ 成功编辑 {len(images)} 张图像!")
       241
       242                            for idx, image in enumerate(images):
       243 -                              st.image(image, caption=f"Edited Image
           -  {idx + 1}", use_container_width=True)
       243 +                              st.image(image, caption=f"编辑后图像 {idx + 1}",
     use_container_width=True)
       244
       245                                buf = io.BytesIO()
       246                                image.save(buf, format='PNG')
       247                                buf.seek(0)
       248
       249                                st.download_button(
       250 -                                  label=f"⬇️ Download Edited Image {idx + 1}",
       250 +                                  label=f"⬇️ 下载编辑后图像 {idx + 1}",
       251                                    data=buf,
       252                                    file_name=f"longcat_edit_{idx + 1}.png",
       253                                    mime="image/png",
     ...
       255                                )
       256
       257                    except Exception as e:
       258 -                      st.error(f"❌ Error editing image: {str(e)}")
       258 +                      st.error(f"❌ 编辑图像时出错: {str(e)}")
       259                        st.exception(e)
       260                else:
       261 -                  st.info("👈 Upload an image and configure parameters to start editing")
       261 +                  st.info("👈 上传图像并配置参数开始编辑")
       262
       263        with tab3:
       264 -          st.header("About LongCat-Image")
       264 +          st.header("关于 LongCat-Image")
       265
       266            st.markdown("""
       267 -          ### 🌟 Key Features
       267 +          ### 🌟 核心特性
       268
       269 -          - **Exceptional Efficiency and Performance**: With only **6B parameters**, LongCat-Image
           - surpasses numerous open-source models that are several times larger across multiple benchmarks.
       269 +          - **卓越的效率与性能**: 仅用 **6B 参数**, LongCat-Image
           + 在多个基准测试中超越了许多体积数倍的开源模型。
       270
       271 -          - **Superior Editing Performance**: LongCat-Image-Edit model achieves state-of-the-art
           - performance among open-source models with superior visual consistency.
       271 +          - **出色的编辑性能**: LongCat-Image-Edit
           + 模型在开源模型中实现了最先进的性能,具有卓越的视觉一致性。
       272
       273 -          - **Powerful Chinese Text Rendering**: Demonstrates superior accuracy and stability in
           - rendering common Chinese characters compared to existing SOTA open-source models.
       273 +          - **强大的中文文字渲染**: 在常见中文字符渲染方面,相比现有 SOTA
           + 开源模型表现出卓越的准确性和稳定性。
       274
       275 -          - **Remarkable Photorealism**: Through an innovative data strategy and training framework,
           - LongCat-Image achieves remarkable photorealism in generated images.
       275 +          - **出色的照片真实感**: 通过创新的数据策略和训练框架, LongCat-Image
           + 在生成图像中实现了出色的照片真实感。
       276
       277 -          - **Comprehensive Open-Source Ecosystem**: Complete toolchain from intermediate checkpoints
           -  to full training code.
       277 +          - **全面的开源生态系统**: 从中间检查点到完整训练代码的完整工具链。
       278
       279 -          ### 📚 Resources
       279 +          ### 📚 资源
       280
       281 -          - [GitHub Repository](https://github.com/meituan-longcat/LongCat-Image)
       282 -          - [Technical Report on arXiv](https://arxiv.org/pdf/2512.07584)
       283 -          - [Online Demo](https://longcat.ai/)
       281 +          - [GitHub 仓库](https://github.com/meituan-longcat/LongCat-Image)
       282 +          - [arXiv 技术报告](https://arxiv.org/pdf/2512.07584)
       283 +          - [在线演示](https://longcat.ai/)
       284            - [Hugging Face - LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image)
       285            - [Hugging Face -
             LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit)
       286
       287 -          ### 📝 Citation
       287 +          ### 📝 引用
       288
       289            ```bibtex
       290            @article{LongCat-Image,
     ...
       295            }
       296            ```
       297
       298 -          ### 📧 Contact
       298 +          ### 📧 联系方式
       299
       300 -          - Email: longcat-team@meituan.com
       300 +          - 邮箱: longcat-team@meituan.com
       301            - Twitter: [@Meituan_LongCat](https://x.com/Meituan_LongCat)
       302
       303 -          ### ⚖️ License
       303 +          ### ⚖️ 许可证
       304
       305 -          LongCat-Image is licensed under Apache 2.0.
       305 +          LongCat-Image 采用 Apache 2.0 许可证。
       306
       307            ---
       308
       309 -          Built with ❤️ by Meituan LongCat Team
       309 +          由美团 LongCat 团队用 ❤️ 构建
       310            """)
       311
       312    if __name__ == "__main__":

● 现在让我也更新中文版的README文档:

● Write(STREAMLIT_README_CN.md)
  ⎿  Wrote 166 lines to STREAMLIT_README_CN.md
     # LongCat-Image Streamlit 网页界面
     一个用户友好的 LongCat-Image 网页界面,使用 Streamlit 构建,支持文生图和图像编辑功能。
     ## 功能特性
     - **文生图生成**: 从中英文文本提示词生成高质量图像
     - **图像编辑**: 使用自然语言指令编辑现有图像
     - **双语支持**: 完整支持中文和英文文字渲染
     - **用户友好界面**: 直观的网页界面,支持实时参数调整
     … +156 lines (ctrl+o to expand)

● 现在让我更新启动脚本为中文:

● Write(run_app.sh)
  ⎿  Updated run_app.sh with 5 additions and 5 removals
        1    #!/bin/bash
        2
        3 -  echo "Starting LongCat-Image Streamlit Web Interface..."
        3 +  echo "启动 LongCat-Image Streamlit 网页界面..."
        4    echo "=================================================="
        5    echo ""
        6 -  echo "Make sure you have:"
        7 -  echo "1. Installed all dependencies: pip install -r requirements.txt"
        8 -  echo "2. Downloaded the models to ./weights/ directory"
        6 +  echo "请确保您已经:"
        7 +  echo "1. 安装了所有依赖: pip install -r requirements.txt"
        8 +  echo "2. 将模型下载到 ./weights/ 目录"
        9    echo ""
       10 -  echo "The app will open in your default browser at http://localhost:8501"
       10 +  echo "应用将在您的默认浏览器中打开 http://localhost:8501"
       11    echo ""
       12
       13    streamlit run app.py

● Bash(chmod +x /home/ubuntu/github/LongCat-Image/run_app.sh)
  ⎿  (No content)

● 完成!我已经为 LongCat-Image 创建了一个用户友好的中文 Streamlit 网页界面。以下是创建的文件:

  创建的文件

  1. app.py - 主要的 Streamlit 应用程序
    - 支持文生图(Text-to-Image)生成
    - 支持图像编辑(Image Editing)
    - 完全中文界面
    - 包含三个标签页:文生图、图像编辑、关于
  2. STREAMLIT_README_CN.md - 中文使用文档
    - 详细的安装说明
    - 使用指南
    - 参数说明
    - 故障排除
  3. run_app.sh - 启动脚本
    - 一键启动应用
    - 带有中文提示信息
  4. requirements.txt - 已更新,添加了 streamlit 依赖

  主要功能

  文生图功能

  - 支持中英文提示词输入
  - 可调整参数:宽度、高度、引导强度、推理步数
  - 支持 CFG 重归一化和提示词重写
  - 批量生成多张图像
  - 可设置随机种子
  - 一键下载生成的图像

  图像编辑功能

  - 上传图像进行编辑
  - 使用自然语言描述编辑指令
  - 支持负面提示词
  - 参数可调节
  - 批量编辑
  - 下载编辑后的图像

  其他特性

  - 模型路径可配置
  - CPU 卸载选项(节省显存)
  - 响应式布局
  - 实时进度显示
  - 错误处理和异常显示
  - 关于页面展示项目信息

  使用方法

  1. 安装依赖:
  pip install -r requirements.txt
  python setup.py develop

  2. 下载模型:
  huggingface-cli download meituan-longcat/LongCat-Image --local-dir ./weights/LongCat-Image
  huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir ./weights/LongCat-Image-Edit

  3. 启动应用:
  streamlit run app.py
  # 或
  ./run_app.sh

  应用将在浏览器中打开,地址为 http://localhost:8501。
