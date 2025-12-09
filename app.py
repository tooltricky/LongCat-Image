import streamlit as st
import torch
from PIL import Image
import io
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline

st.set_page_config(
    page_title="LongCat-Image 网页界面",
    page_icon="🐱",
    layout="wide"
)

@st.cache_resource
def load_t2i_model(checkpoint_dir, use_cpu_offload=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_processor = AutoProcessor.from_pretrained(checkpoint_dir, subfolder='tokenizer')
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        checkpoint_dir,
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to(device)

    pipe = LongCatImagePipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16
    )

    if use_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, torch.bfloat16)

    return pipe

@st.cache_resource
def load_edit_model(checkpoint_dir, use_cpu_offload=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_processor = AutoProcessor.from_pretrained(checkpoint_dir, subfolder='tokenizer')
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        checkpoint_dir,
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to(device)

    pipe = LongCatImageEditPipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16
    )

    if use_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, torch.bfloat16)

    return pipe

def main():
    st.title("🐱 LongCat-Image 网页界面")
    st.markdown("### 中英双语文生图与图像编辑")

    st.sidebar.header("模型配置")

    t2i_checkpoint = st.sidebar.text_input(
        "文生图模型路径",
        value="./weights/LongCat-Image",
        help="LongCat-Image 模型检查点目录路径"
    )

    edit_checkpoint = st.sidebar.text_input(
        "图像编辑模型路径",
        value="./weights/LongCat-Image-Edit",
        help="LongCat-Image-Edit 模型检查点目录路径"
    )

    use_cpu_offload = st.sidebar.checkbox(
        "启用 CPU 卸载",
        value=True,
        help="启用可节省显存(速度较慢但避免显存溢出)。在高显存设备上禁用可获得更快的推理速度。"
    )

    tab1, tab2, tab3 = st.tabs(["📝 文生图", "✏️ 图像编辑", "ℹ️ 关于"])

    with tab1:
        st.header("文生图生成")
        st.info("⚠️ **文字渲染特殊处理**: 当生成包含文字的图像时,请将目标文字用引号(\"\")括起来以获得更好的质量。")

        col1, col2 = st.columns([1, 1])

        with col1:
            prompt = st.text_area(
                "提示词",
                value='一个年轻的亚裔女性,身穿黄色针织衫,搭配白色项链。她的双手放在膝盖上,表情恬静。背景是一堵粗糙的砖墙,午后的阳光温暖地洒在她身上,营造出一种宁静而温馨的氛围。',
                height=150,
                help="输入中文或英文的图像生成提示词"
            )

            negative_prompt = st.text_area(
                "负面提示词(可选)",
                value='',
                height=80,
                help="描述您不希望在图像中出现的内容"
            )

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                width = st.slider("宽度", min_value=512, max_value=2048, value=1344, step=64)
                guidance_scale = st.slider("引导强度", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
                enable_cfg_renorm = st.checkbox("启用 CFG 重归一化", value=True)

            with col1_2:
                height = st.slider("高度", min_value=512, max_value=2048, value=768, step=64)
                num_inference_steps = st.slider("推理步数", min_value=10, max_value=100, value=50, step=5)
                enable_prompt_rewrite = st.checkbox("启用提示词重写", value=True, help="使用内置的文本编码器作为提示词改写器")

            num_images = st.number_input("图像数量", min_value=1, max_value=4, value=1)
            seed = st.number_input("随机种子", min_value=-1, max_value=999999, value=43, help="使用 -1 表示随机种子")

            generate_button = st.button("🎨 生成图像", type="primary", use_container_width=True)

        with col2:
            if generate_button:
                try:
                    with st.spinner("正在加载模型..."):
                        pipe = load_t2i_model(t2i_checkpoint, use_cpu_offload)

                    with st.spinner("正在生成图像... 这可能需要一段时间。"):
                        generator = torch.Generator("cpu").manual_seed(seed) if seed >= 0 else None

                        result = pipe(
                            prompt,
                            negative_prompt=negative_prompt if negative_prompt else '',
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=num_images,
                            generator=generator,
                            enable_cfg_renorm=enable_cfg_renorm,
                            enable_prompt_rewrite=enable_prompt_rewrite
                        )

                        images = result.images

                        st.success(f"✅ 成功生成 {len(images)} 张图像!")

                        for idx, image in enumerate(images):
                            st.image(image, caption=f"生成图像 {idx + 1}", use_container_width=True)

                            buf = io.BytesIO()
                            image.save(buf, format='PNG')
                            buf.seek(0)

                            st.download_button(
                                label=f"⬇️ 下载图像 {idx + 1}",
                                data=buf,
                                file_name=f"longcat_t2i_{idx + 1}.png",
                                mime="image/png",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ 生成图像时出错: {str(e)}")
                    st.exception(e)
            else:
                st.info("👈 配置参数后点击'生成图像'开始")

    with tab2:
        st.header("图像编辑")
        st.info("⚠️ **文字渲染特殊处理**: 当编辑包含文字的图像时,请将目标文字用引号(\"\")括起来以获得更好的质量。")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "上传图像",
                type=['png', 'jpg', 'jpeg'],
                help="上传要编辑的图像"
            )

            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert('RGB')
                st.image(input_image, caption="输入图像", use_container_width=True)

            edit_prompt = st.text_area(
                "编辑指令",
                value='将猫变成狗',
                height=100,
                help="描述您想如何编辑图像"
            )

            edit_negative_prompt = st.text_area(
                "负面提示词(可选)",
                value='',
                height=80,
                help="描述您不希望在编辑后的图像中出现的内容"
            )

            col2_1, col2_2 = st.columns(2)
            with col2_1:
                edit_guidance_scale = st.slider("引导强度", min_value=1.0, max_value=10.0, value=4.5, step=0.1, key="edit_guidance")
                edit_num_images = st.number_input("图像数量", min_value=1, max_value=4, value=1, key="edit_num_images")

            with col2_2:
                edit_num_inference_steps = st.slider("推理步数", min_value=10, max_value=100, value=50, step=5, key="edit_steps")
                edit_seed = st.number_input("随机种子", min_value=-1, max_value=999999, value=43, help="使用 -1 表示随机种子", key="edit_seed")

            edit_button = st.button("✏️ 编辑图像", type="primary", use_container_width=True, disabled=(uploaded_file is None))

        with col2:
            if edit_button and uploaded_file is not None:
                try:
                    with st.spinner("正在加载模型..."):
                        edit_pipe = load_edit_model(edit_checkpoint, use_cpu_offload)

                    with st.spinner("正在编辑图像... 这可能需要一段时间。"):
                        edit_generator = torch.Generator("cpu").manual_seed(edit_seed) if edit_seed >= 0 else None

                        result = edit_pipe(
                            input_image,
                            edit_prompt,
                            negative_prompt=edit_negative_prompt if edit_negative_prompt else '',
                            guidance_scale=edit_guidance_scale,
                            num_inference_steps=edit_num_inference_steps,
                            num_images_per_prompt=edit_num_images,
                            generator=edit_generator
                        )

                        images = result.images

                        st.success(f"✅ 成功编辑 {len(images)} 张图像!")

                        for idx, image in enumerate(images):
                            st.image(image, caption=f"编辑后图像 {idx + 1}", use_container_width=True)

                            buf = io.BytesIO()
                            image.save(buf, format='PNG')
                            buf.seek(0)

                            st.download_button(
                                label=f"⬇️ 下载编辑后图像 {idx + 1}",
                                data=buf,
                                file_name=f"longcat_edit_{idx + 1}.png",
                                mime="image/png",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ 编辑图像时出错: {str(e)}")
                    st.exception(e)
            else:
                st.info("👈 上传图像并配置参数开始编辑")

    with tab3:
        st.header("关于 LongCat-Image")

        st.markdown("""
        ### 🌟 核心特性

        - **卓越的效率与性能**: 仅用 **6B 参数**, LongCat-Image 在多个基准测试中超越了许多体积数倍的开源模型。

        - **出色的编辑性能**: LongCat-Image-Edit 模型在开源模型中实现了最先进的性能,具有卓越的视觉一致性。

        - **强大的中文文字渲染**: 在常见中文字符渲染方面,相比现有 SOTA 开源模型表现出卓越的准确性和稳定性。

        - **出色的照片真实感**: 通过创新的数据策略和训练框架, LongCat-Image 在生成图像中实现了出色的照片真实感。

        - **全面的开源生态系统**: 从中间检查点到完整训练代码的完整工具链。

        ### 📚 资源

        - [GitHub 仓库](https://github.com/meituan-longcat/LongCat-Image)
        - [arXiv 技术报告](https://arxiv.org/pdf/2512.07584)
        - [在线演示](https://longcat.ai/)
        - [Hugging Face - LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image)
        - [Hugging Face - LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit)

        ### 📝 引用

        ```bibtex
        @article{LongCat-Image,
              title={LongCat-Image Technical Report},
              author={Meituan LongCat Team and  Hanghang Ma and Haoxian Tan and Jiale Huang and Junqiang Wu and Jun-Yan He and Lishuai Gao and Songlin Xiao and Xiaoming Wei and Xiaoqi Ma and Xunliang Cai and Yayong Guan and Jie Hu},
              journal={arXiv preprint arXiv:2512.07584},
              year={2025}
        }
        ```

        ### 📧 联系方式

        - 邮箱: longcat-team@meituan.com
        - Twitter: [@Meituan_LongCat](https://x.com/Meituan_LongCat)

        ### ⚖️ 许可证

        LongCat-Image 采用 Apache 2.0 许可证。

        ---

        由美团 LongCat 团队用 ❤️ 构建
        """)

if __name__ == "__main__":
    main()
