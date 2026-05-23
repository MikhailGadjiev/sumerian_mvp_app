"""
Gradio MVP for the SumerianTranslator project.

What it does:
1. Accepts an image of a cuneiform tablet / lineart.
2. Optionally runs YOLO detection if YOLO_MODEL and YOLO_DATA_YAML are configured.
3. Accepts or edits transliteration.
4. Optionally runs ByT5 translation if TRANSLATION_MODEL_DIR is configured.

This file is intentionally safe for a defense demo: if model weights are absent, it still runs
in demo mode and clearly shows which components are not connected yet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw


YOLO_MODEL = os.getenv("YOLO_MODEL", "").strip()
YOLO_DATA_YAML = os.getenv("YOLO_DATA_YAML", "").strip()
TRANSLATION_MODEL_DIR = os.getenv("TRANSLATION_MODEL_DIR", "").strip()
SOURCE_PREFIX = os.getenv("SOURCE_PREFIX", "Переведи шумерский на русский: ")


def _draw_demo_boxes(image: Image.Image) -> Image.Image:
    """Fallback visualization when real OCR weights are not configured."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    boxes = [
        (int(w * 0.15), int(h * 0.20), int(w * 0.32), int(h * 0.34)),
        (int(w * 0.38), int(h * 0.20), int(w * 0.55), int(h * 0.34)),
        (int(w * 0.61), int(h * 0.20), int(w * 0.78), int(h * 0.34)),
    ]
    labels = ["𒀭", "𒈗", "𒆠"]
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 18)), label, fill="red")
    return img


def run_ocr(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], str, str]:
    """Run OCR/detection if weights exist; otherwise demo fallback."""
    if image is None:
        return None, "", "Загрузите изображение."

    if YOLO_MODEL and Path(YOLO_MODEL).exists():
        try:
            from ultralytics import YOLO

            model = YOLO(YOLO_MODEL)
            results = model.predict(image, verbose=False)
            annotated = Image.fromarray(results[0].plot()[:, :, ::-1])

            # This is a simple placeholder: true reading-order decoding depends on your CER.py logic.
            transliteration = "[результат OCR: требуется подключить decoding / reading_order из репозитория detection]"
            status = f"YOLO-модель загружена: {YOLO_MODEL}"
            return annotated, transliteration, status
        except Exception as e:
            annotated = _draw_demo_boxes(image)
            return annotated, "", f"Не удалось запустить реальную OCR-модель: {e}"

    annotated = _draw_demo_boxes(image)
    demo_translit = "an lugal ki"
    status = (
        "Демо-режим: веса OCR не подключены. "
        "Укажите переменную окружения YOLO_MODEL=/path/to/best.pt для реального запуска."
    )
    return annotated, demo_translit, status


_translation_cache = {"model": None, "tokenizer": None, "error": None}


def _load_translation_model():
    if _translation_cache["model"] is not None or _translation_cache["error"] is not None:
        return _translation_cache["model"], _translation_cache["tokenizer"], _translation_cache["error"]

    if not TRANSLATION_MODEL_DIR or not Path(TRANSLATION_MODEL_DIR).exists():
        _translation_cache["error"] = (
            "Демо-режим: модель перевода не подключена. "
            "Укажите TRANSLATION_MODEL_DIR=/path/to/saved/byt5_model."
        )
        return None, None, _translation_cache["error"]

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _translation_cache["model"] = model
        _translation_cache["tokenizer"] = tokenizer
        return model, tokenizer, None
    except Exception as e:
        _translation_cache["error"] = str(e)
        return None, None, str(e)


def translate(transliteration: str) -> Tuple[str, str]:
    text = (transliteration or "").strip()
    if not text:
        return "", "Введите транслитерацию."

    model, tokenizer, error = _load_translation_model()
    if model is None or tokenizer is None:
        # Fallback demonstration so the interface works without private weights.
        return (
            "[демо-перевод] Предварительный перевод будет сформирован моделью ByT5 после подключения весов.",
            error or "Модель перевода не подключена.",
        )

    try:
        import torch

        device = next(model.parameters()).device
        encoded = tokenizer(
            [SOURCE_PREFIX + text],
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
        result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return result, f"Модель перевода загружена: {TRANSLATION_MODEL_DIR}"
    except Exception as e:
        return "", f"Ошибка перевода: {e}"


def full_pipeline(image: Optional[Image.Image], manual_transliteration: str) -> Tuple[Optional[Image.Image], str, str, str]:
    annotated, ocr_translit, ocr_status = run_ocr(image)
    final_translit = (manual_transliteration or "").strip() or ocr_translit
    translation, tr_status = translate(final_translit)
    return annotated, final_translit, translation, f"OCR: {ocr_status}\nПеревод: {tr_status}"


with gr.Blocks(title="Sumerian Translator MVP") as demo:
    gr.Markdown(
        "# Sumerian Translator MVP\n"
        "Модульный прототип: изображение → OCR → транслитерация → перевод → ручная проверка."
    )

    with gr.Tab("Полный пайплайн"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Изображение клинописной таблички / автографии")
            annotated_out = gr.Image(type="pil", label="Результат OCR / детекции")
        manual_translit = gr.Textbox(label="Ручная транслитерация / корректировка", lines=3)
        run_btn = gr.Button("Запустить пайплайн")
        translit_out = gr.Textbox(label="Итоговая транслитерация", lines=3)
        translation_out = gr.Textbox(label="Предварительный русский перевод", lines=5)
        status_out = gr.Textbox(label="Статус модулей", lines=4)
        run_btn.click(
            fn=full_pipeline,
            inputs=[image_in, manual_translit],
            outputs=[annotated_out, translit_out, translation_out, status_out],
        )

    with gr.Tab("Только перевод"):
        translit_in = gr.Textbox(label="Шумерская транслитерация", lines=5, value="an lugal ki")
        translate_btn = gr.Button("Перевести")
        translation_only = gr.Textbox(label="Русский перевод", lines=5)
        translation_status = gr.Textbox(label="Статус", lines=2)
        translate_btn.click(fn=translate, inputs=translit_in, outputs=[translation_only, translation_status])

    with gr.Tab("Справка"):
        gr.Markdown(
            "## Как подключить реальные модели\n"
            "Перед запуском задайте переменные окружения:\n\n"
            "```bash\n"
            "export YOLO_MODEL=/path/to/best.pt\n"
            "export YOLO_DATA_YAML=/path/to/dataset.yaml\n"
            "export TRANSLATION_MODEL_DIR=/path/to/saved/byt5_model\n"
            "python app.py\n"
            "```\n\n"
            "Если веса не указаны, приложение запускается в демо-режиме. "
            "Это удобно для презентации архитектуры, но для реального инференса нужны сохранённые веса моделей."
        )


if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False,
    inbrowser=True,
    prevent_thread_lock=False
)
