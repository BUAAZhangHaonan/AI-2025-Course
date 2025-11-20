# -*- coding: utf-8 -*-
import os
import gc
import torch
from PIL import Image
import gradio as gr
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MATCH_CKPT_BASE  = "model_base_retrieval_flickr.pth"#https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth
MATCH_CKPT_LARGE = "model_large_retrieval_flickr.pth"#https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth


MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
STD_CLIP  = (0.26862954, 0.26130258, 0.27577711)

def build_preprocess(sz: int):
    return transforms.Compose([
        transforms.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIP, std=STD_CLIP),
    ])


SIZE_MATCH = 384
SIZE_VQA   = 480
preprocess_match = build_preprocess(SIZE_MATCH)
preprocess_vqa   = build_preprocess(SIZE_VQA)


MODEL_CACHE = {}

def _release_model(key):
    entry = MODEL_CACHE.get(key)
    if entry is not None:
        _, model = entry
        try:
            del MODEL_CACHE[key]
            if model is not None:
                del model
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_or_load_model(task_key: str, ckpt_path: str):
    entry = MODEL_CACHE.get(task_key)
    if entry is not None:
        prev_path, model = entry
        if os.path.abspath(prev_path) == os.path.abspath(ckpt_path):
            return model
        else:
            _release_model(task_key)

    # 真正加载
    if task_key in ("match_base", "match_large"):
        vit = "base" if task_key == "match_base" else "large"
        model = blip_itm(pretrained=ckpt_path, image_size=SIZE_MATCH, vit=vit)
    elif task_key in ("vqa_capfilt", "vqa_plain"):
        model = blip_vqa(pretrained=ckpt_path, image_size=SIZE_VQA, vit="base")
    else:
        raise ValueError(f"Unknown task key: {task_key}")

    model.eval()
    model = model.to(DEVICE)
    MODEL_CACHE[task_key] = (ckpt_path, model)
    return model


@torch.inference_mode()
def run_infer(task,
              image_pil,
              text_or_question,
              vit_choice,
              vqa_weight_choice,
              ckpt_vqa_capfilt,
              ckpt_vqa_plain):

    if image_pil is None:
        return "请先上传图片。", None

    img = image_pil.convert("RGB")


    if task == "match":
        if not text_or_question or not text_or_question.strip():
            return "请输入待匹配文本。", None

        if vit_choice == "ViT-Base":
            ckpt_path = MATCH_CKPT_BASE
            task_key = "match_base"
            vit_tag = "base"
        else:
            ckpt_path = MATCH_CKPT_LARGE
            task_key = "match_large"
            vit_tag = "large"

        image_tensor = preprocess_match(img).unsqueeze(0).to(DEVICE)
        model = get_or_load_model(task_key, ckpt_path)


        itm_output = model(image_tensor, text_or_question.strip(), match_head='itm')
        itm_prob = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()

        itc_score = model(image_tensor, text_or_question.strip(), match_head='itc').item()

        out = (f"[Match] ViT: {vit_tag}\n"
               f"Text: {text_or_question.strip()}\n"
               f"ITM prob (matched): {itm_prob:.4f}\n"
               f"ITC cosine similarity: {itc_score:.4f}")
        return out, image_pil


    else:
        if not text_or_question or not text_or_question.strip():
            return "请输入问题（Question）。", None

        if vqa_weight_choice == "capfilt":
            ckpt_path = (ckpt_vqa_capfilt or "").strip()
            if not ckpt_path:
                return "请填写 VQA capfilt 权重路径。", None
            task_key = "vqa_capfilt"
            weight_tag = "capfilt_large"
        else:
            ckpt_path = (ckpt_vqa_plain or "").strip()
            if not ckpt_path:
                return "请填写 VQA 普通 vqa 权重路径。", None
            task_key = "vqa_plain"
            weight_tag = "vqa"

        image_tensor = preprocess_vqa(img).unsqueeze(0).to(DEVICE)
        model = get_or_load_model(task_key, ckpt_path)

        ans = model(image_tensor, text_or_question.strip(), train=False, inference="generate")
        ans_text = ans[0] if isinstance(ans, (list, tuple)) else str(ans)

        out = (f"[VQA] ViT: base | Weights: {weight_tag}\n"
               f"Q: {text_or_question.strip()}\n"
               f"A: {ans_text}")
        return out, image_pil


def build_ui():
    with gr.Blocks(title="BLIP Match / VQA 评测（Match 固定 ckpt；VQA 双权重）", analytics_enabled=False) as demo:
        gr.Markdown("## 🧠 BLIP Match / VQA 评测\n"
                    "- 先选择 **任务：match / vqa**。\n"
                    )

        with gr.Row():
            with gr.Column():
                task = gr.Radio(["match", "vqa"], value="match", label="任务")

                # match：仅选择 ViT
                vit_choice = gr.Dropdown(["ViT-Base", "ViT-Large"], value="ViT-Base",
                                         label="（match）选择 ViT", visible=True)

                img_in = gr.Image(type="pil", label="上传图片", sources=["upload"])
                text_in = gr.Textbox(label="Text（match）或 Question（vqa）",
                                     placeholder="match 请输入待匹配文本；vqa 请输入问题。")

            with gr.Column():
                gr.Markdown("### vqa（问答，ViT-Base，两权重可切换）")
                vqa_weight_choice = gr.Radio(
                    choices=["capfilt", "plain"],
                    value="capfilt",
                    label="选择 VQA 权重",
                    visible=False
                )
                ckpt_vqa_capfilt = gr.Textbox(
                    label="vqa capfilt_large ckpt 路径",
                    value="model_base_vqa_capfilt_large.pth",#https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth
                    visible=False
                )
                ckpt_vqa_plain = gr.Textbox(
                    label="vqa 普通版 ckpt 路径",
                    value="model_vqa.pth",#https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth
                    visible=False
                )

            with gr.Column():
                btn = gr.Button("🚀 运行", variant="primary")
                out_text = gr.Textbox(label="输出", lines=8)
                out_img = gr.Image(type="pil", label="输入图回显")

        # 动态显隐：match 显示 vit_choice；vqa 显示 vqa 权重与路径
        def _toggle_fields(task_choice):
            return (
                gr.update(visible=(task_choice == "match")),  # vit_choice
                gr.update(visible=(task_choice == "vqa")),    # vqa_weight_choice
                gr.update(visible=(task_choice == "vqa")),    # ckpt_vqa_capfilt
                gr.update(visible=(task_choice == "vqa")),    # ckpt_vqa_plain
            )

        task.change(
            _toggle_fields,
            inputs=[task],
            outputs=[vit_choice, vqa_weight_choice, ckpt_vqa_capfilt, ckpt_vqa_plain]
        )

        btn.click(
            fn=run_infer,
            inputs=[
                task, img_in, text_in, vit_choice,
                vqa_weight_choice, ckpt_vqa_capfilt, ckpt_vqa_plain
            ],
            outputs=[out_text, out_img]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
