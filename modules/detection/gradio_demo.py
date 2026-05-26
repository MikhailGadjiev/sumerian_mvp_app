"""
Gradio UI for YOLO Sumerian cuneiform detection.

Run from repository root (so paths in CER.py resolve correctly), for example:
  uv run python models/yolo/gradio_demo.py
  uv run python models/yolo/gradio_demo.py --model /path/to/best.pt --data /path/to/dataset.yaml
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = Path(__file__).resolve().parent

from CER import CER, reading_order  # noqa: E402


def load_class_names(dataset_yaml: Path) -> list[str]:
    with open(dataset_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data["names"])


def glyph_catalog_choices(names: list[str]) -> list[tuple[str, str]]:
    """Уникальные знаки из names (как в dataset.yaml); для выпадающего списка (label, value)."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for raw in names:
        g = raw.replace("<OTHER>", "?").strip()
        if not g or g in seen:
            continue
        seen.add(g)
        rows.append((g, g))
    return sorted(rows, key=lambda t: t[0])


def lineart_context_dict(
    w: int,
    h: int,
    fb: list,
    ft: list[str],
    fconf: list[float],
    reading_alpha: float,
) -> dict[str, Any]:
    """Состояние для правок автографии по bbox (без повторного YOLO)."""
    return {
        "w": int(w),
        "h": int(h),
        "fb": list(fb),
        "ft": list(ft),
        "fconf": list(fconf),
        "reading_alpha": float(reading_alpha),
    }


def save_numpy_image_temp(arr: np.ndarray | None, fmt: str, stem: str) -> str | None:
    """Пишет массив RGB в временный файл PNG или JPEG; путь для gr.File."""
    if arr is None:
        return None
    fmt_l = fmt.lower()
    if fmt_l in ("jpg", "jpeg"):
        pil_fmt, suffix = "JPEG", ".jpg"
    elif fmt_l == "png":
        pil_fmt, suffix = "PNG", ".png"
    else:
        pil_fmt, suffix = "PNG", ".png"

    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if pil_fmt == "JPEG" and pil.mode in ("RGBA", "P"):
        pil = pil.convert("RGB")

    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{stem}_")
    os.close(fd)
    kw: dict = {}
    if pil_fmt == "JPEG":
        kw["quality"] = 92
        kw["optimize"] = True
    pil.save(path, format=pil_fmt, **kw)
    return path


def resolve_font(font_arg: str | None) -> tuple[Path | None, str | None]:
    candidates: list[Path] = []
    if font_arg:
        candidates.append(Path(font_arg).expanduser())
    candidates.extend(
        [
            YOLO_DIR / "NotoSansCuneiform-Regular.ttf",
            Path("/usr/share/fonts/truetype/noto/NotoSansCuneiform-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCuneiform-Regular.otf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCuneiform-Regular.otf"),
        ]
    )
    for p in candidates:
        if p.is_file():
            return p, None
    return None, (
        "Шрифт Noto Sans Cuneiform не найден. Подписи к боксам могут отображаться как квадраты. "
        "Укажите путь через --font или положите NotoSansCuneiform-Regular.ttf в каталог models/yolo."
    )


def filter_bboxes_texts_confs(
    bboxes: list,
    texts: list[str],
    confs: list[float],
) -> tuple[list, list[str], list[float]]:
    """Та же логика, что CER.filter_bboxes, плюс отфильтрованные confidences."""
    assert len(bboxes) == len(texts) == len(confs)
    filtered_bboxes: list = []
    filtered_texts: list[str] = []
    filtered_confs: list[float] = []
    for i, (box_i, text_i, conf_i) in enumerate(zip(bboxes, texts, confs)):
        x1_i, y1_i, x2_i, y2_i = box_i
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        inside = False
        for j, box_j in enumerate(bboxes):
            if i == j:
                continue
            x1_j, y1_j, x2_j, y2_j = box_j
            inter_area = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) * max(
                0, min(y2_i, y2_j) - max(y1_i, y1_j)
            )
            if area_i > 0 and inter_area / area_i >= 0.8:
                inside = True
                break
        if not inside:
            filtered_bboxes.append(box_i)
            filtered_texts.append(text_i)
            filtered_confs.append(conf_i)
    return filtered_bboxes, filtered_texts, filtered_confs


def _reading_order_safe(xyxy: list, texts: list[str], alpha: float) -> str:
    if not xyxy:
        return ""
    try:
        return reading_order(xyxy, texts, alpha=alpha)
    except (ValueError, IndexError):
        return ""


def render_lineart_autography(
    src_w: int,
    src_h: int,
    fb: list[list[float]],
    ft: list[str],
    reading_alpha: float,
    font_path: Path | None,
    canvas_max_width: int = 1400,
    *,
    box_confs: list[float] | None = None,
    show_confidence: bool = False,
    bbox_overlay: bool = False,
    bbox_highlight_index: int | None = None,
) -> np.ndarray | None:
    """
    Реконструкция «автографии»: белый фон, чёрные глифы в позициях детекций,
    только горизонтальные линии регистров (DBSCAN по Y, ``reading_alpha`` как у порядка чтения).

    При ``show_confidence=True`` под каждым знаком рисуется значение уверенности (только для превью).
    Экспорт для учёных — с ``show_confidence=False``.

    При ``bbox_overlay=True`` вокруг каждого знака рисуется рамка в координатах автографии;
    ``bbox_highlight_index`` задаёт рамку активного знака (для выбора в UI).
    """
    if not fb or not ft:
        return None

    scale = canvas_max_width / float(src_w)
    w_out = int(canvas_max_width)
    h_out = max(1, int(round(src_h * scale)))

    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in fb], dtype=np.float64)
    heights = np.array([b[3] - b[1] for b in fb], dtype=np.float64)
    median_h = float(np.median(heights)) if len(heights) else 1.0
    eps = median_h * float(reading_alpha)
    ys = centers[:, 1].reshape(-1, 1)
    line_ids = DBSCAN(eps=eps, min_samples=1).fit(ys).labels_
    line_order = sorted(set(line_ids), key=lambda lid: float(np.mean(ys[line_ids == lid])))

    row_bounds: list[tuple[float, float]] = []
    for lid in line_order:
        idxs = np.where(line_ids == lid)[0]
        tops = [fb[int(i)][1] for i in idxs]
        bottoms = [fb[int(i)][3] for i in idxs]
        row_bounds.append((min(tops), max(bottoms)))

    canvas = Image.new("RGB", (w_out, h_out), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    pad_src = median_h * 0.12
    all_x1 = min(b[0] for b in fb) - pad_src
    all_y1 = min(b[1] for b in fb) - pad_src
    all_x2 = max(b[2] for b in fb) + pad_src
    all_y2 = max(b[3] for b in fb) + pad_src
    frame_sw = max(1, int(round(2 * scale)))
    draw.rectangle(
        [
            int(all_x1 * scale),
            int(all_y1 * scale),
            int(all_x2 * scale),
            int(all_y2 * scale),
        ],
        outline=(0, 0, 0),
        width=frame_sw,
    )

    xL = int(round(all_x1 * scale))
    xR = int(round(all_x2 * scale))

    for i, bbox in enumerate(fb):
        x1, y1, x2, y2 = bbox
        sx1, sy1 = x1 * scale, y1 * scale
        sx2, sy2 = x2 * scale, y2 * scale
        cx = (sx1 + sx2) / 2.0
        cy = (sy1 + sy2) / 2.0
        box_h = max(1.0, sy2 - sy1)
        font_size = int(max(14, min(128, box_h * 0.88)))
        try:
            if font_path is not None:
                font = ImageFont.truetype(str(font_path), size=font_size)
            else:
                font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()

        label = ft[i]
        # Якорь "mm" (middle-middle): центр чернил совпадает с геометрическим центром бокса.
        # Ручной расчёт через textbbox((0,0)) + text((tx,ty)) даёт систематический сдвиг по Y.
        draw.text((cx, cy), label, font=font, fill=(0, 0, 0), anchor="mm")

        if (
            show_confidence
            and box_confs is not None
            and i < len(box_confs)
        ):
            cfs = float(box_confs[i])
            conf_size = int(max(9, min(26, font_size * 0.30)))
            try:
                if font_path is not None:
                    font_small = ImageFont.truetype(str(font_path), size=conf_size)
                else:
                    font_small = ImageFont.load_default()
            except OSError:
                font_small = ImageFont.load_default()
            gap = max(2.0, float(conf_size) * 0.35)
            draw.text(
                (cx, sy2 + gap),
                f"{cfs:.2f}",
                font=font_small,
                fill=(72, 72, 72),
                anchor="mt",
            )

    for i in range(len(row_bounds) - 1):
        y_sep_src = (row_bounds[i][1] + row_bounds[i + 1][0]) / 2.0
        y_pix = int(round(y_sep_src * scale))
        lw = max(1, int(round(median_h * 0.06 * scale)))
        draw.line([(xL, y_pix), (xR, y_pix)], fill=(0, 0, 0), width=lw)

    if bbox_overlay:
        tag_size = int(max(10, min(22, 14 * scale)))
        try:
            if font_path is not None:
                tag_font = ImageFont.truetype(str(font_path), size=tag_size)
            else:
                tag_font = ImageFont.load_default()
        except OSError:
            tag_font = ImageFont.load_default()
        for i, bbox in enumerate(fb):
            x1, y1, x2, y2 = bbox
            sx1, sy1 = x1 * scale, y1 * scale
            sx2, sy2 = x2 * scale, y2 * scale
            hi = bbox_highlight_index is not None and i == bbox_highlight_index
            rect_col = (255, 115, 0) if hi else (70, 130, 220)
            lw_box = 5 if hi else 2
            draw.rectangle(
                [int(sx1), int(sy1), int(sx2), int(sy2)],
                outline=rect_col,
                width=lw_box,
            )
            tag = f"[{i}]"
            draw.text(
                (sx1 + 3, sy1 + 2),
                tag,
                font=tag_font,
                fill=(180, 30, 0) if hi else (25, 40, 120),
            )

    return np.array(canvas)


def run_inference(
    image: np.ndarray | None,
    names: list[str],
    model: YOLO,
    conf: float,
    iou: float,
    reading_alpha: float,
    ground_truth: str,
    font_path: Path | None,
    static_font_warning: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str, dict[str, Any] | None]:
    warnings: list[str] = []
    if static_font_warning:
        warnings.append(static_font_warning)

    if image is None:
        return None, None, None, "**Загрузите изображение.**", None

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return None, None, None, "Ожидается RGB/RGBA изображение.", None

    rgb = image[:, :, :3].copy()
    h, w = rgb.shape[:2]

    results = model.predict(
        rgb,
        conf=float(conf),
        iou=float(iou),
        agnostic_nms=True,
        verbose=False,
    )
    if not results:
        return rgb, None, None, "Нет результатов от модели.", None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        summary = "**Детекций нет.**"
        if warnings:
            summary = "\n\n".join([*warnings, summary])
        return rgb, None, None, summary, None

    xyxy_list: list[list[float]] = boxes.xyxy.cpu().numpy().tolist()
    conf_list = boxes.conf.cpu().numpy().tolist()
    cls_list = boxes.cls.cpu().numpy().astype(int).tolist()

    texts: list[str] = []
    for c in cls_list:
        if 0 <= c < len(names):
            glyphs = names[c]
        else:
            glyphs = "?"
        texts.append(glyphs.replace("<OTHER>", "?"))

    display_labels = [t.replace("<OTHER>", "?") for t in texts]
    ordered = _reading_order_safe(xyxy_list, display_labels, reading_alpha)

    fb, ft, fconf = filter_bboxes_texts_confs(xyxy_list, display_labels, conf_list)

    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    if font_path is not None:
        font = ImageFont.truetype(str(font_path), size=max(24, min(60, h // 20)))
    else:
        font = ImageFont.load_default()

    for i, bbox in enumerate(fb):
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        label = ft[i]
        ty = max(0, y1 - 4)
        draw.text((x1, ty), label, font=font, fill=(0, 0, 255))

    annotated = np.array(pil.convert("RGB"))
    lineart_preview = render_lineart_autography(
        w,
        h,
        fb,
        ft,
        reading_alpha,
        font_path,
        box_confs=fconf,
        show_confidence=True,
    )
    lineart_export = render_lineart_autography(
        w,
        h,
        fb,
        ft,
        reading_alpha,
        font_path,
        show_confidence=False,
    )

    lines = [
        "### Прочитанная строка (порядок чтения)",
        f"`{ordered}`" if ordered else "`(пусто)`",
        "",
        "### Редактирование автографии",
        "Рамки **bbox** совпадают с позициями знаков на автографии (номер `[i]` в углу). "
        "Выберите знак в списке — активная рамка подсвечивается оранжевым; введите **новый символ** и нажмите "
        "**«Заменить знак на автографии»** (перерисовка без повторного YOLO).",
        "",
        "### Автография / lineart",
        "Чёрные глифы Noto Sans Cuneiform на белом фоне в масштабированных координатах детекции; "
        "линии регистров (только горизонтальные) — по DBSCAN по строкам (тот же `alpha`, что и у порядка чтения). "
        "**Справка:** второй ряд — превью с **conf**; оба превью показывают **одни и те же bbox** (выбранный индекс подсвечен). "
        "Файл для скачивания — автография **без рамок и без чисел conf**.",
        "",
        "### Детекции",
    ]
    for j in range(len(xyxy_list)):
        x1, y1, x2, y2 = xyxy_list[j]
        lines.append(
            f"- `{display_labels[j]}` conf={conf_list[j]:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
        )

    gt = (ground_truth or "").strip()
    if gt:
        cer = CER(ordered, gt)
        lines.extend(["", "### CER (предсказание vs эталон)", f"- Эталон: `{gt}`", f"- **CER: {cer:.4f}**"])

    summary = "\n".join(lines)
    if warnings:
        summary = "\n\n".join(["\n".join(f"- {w}" for w in warnings), summary])

    ctx = lineart_context_dict(w, h, fb, ft, fconf, reading_alpha)
    return annotated, lineart_preview, lineart_export, summary, ctx


def build_demo(
    model_path: Path,
    dataset_yaml: Path,
    font_path: Path | None,
    font_warning: str | None,
) -> gr.Blocks:
    names = load_class_names(dataset_yaml)
    catalog_pairs = glyph_catalog_choices(names)
    model = YOLO(str(model_path))

    def lineart_views(
        ctx: dict[str, Any],
        ft: list[str],
        highlight_i: int | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Чистая автография + bbox; превью с conf + bbox (одинаковое выделение)."""
        if not ctx or not ctx["fb"] or not ft:
            return None, None
        w, h, fb = ctx["w"], ctx["h"], ctx["fb"]
        ra = ctx["reading_alpha"]
        fconf = ctx["fconf"]
        box = render_lineart_autography(
            w,
            h,
            fb,
            ft,
            ra,
            font_path,
            show_confidence=False,
            bbox_overlay=True,
            bbox_highlight_index=highlight_i,
        )
        conf = render_lineart_autography(
            w,
            h,
            fb,
            ft,
            ra,
            font_path,
            box_confs=fconf,
            show_confidence=True,
            bbox_overlay=True,
            bbox_highlight_index=highlight_i,
        )
        return box, conf

    def lineart_clean_only(ctx: dict[str, Any], ft: list[str]) -> np.ndarray | None:
        if not ctx or not ctx["fb"] or not ft:
            return None
        return render_lineart_autography(
            ctx["w"],
            ctx["h"],
            ctx["fb"],
            ft,
            ctx["reading_alpha"],
            font_path,
            show_confidence=False,
        )

    def _predict(image, conf, iou, reading_alpha, ground_truth, download_format):
        annotated, lineart_preview, lineart_export, summary, ctx = run_inference(
            image,
            names,
            model,
            conf,
            iou,
            reading_alpha,
            ground_truth,
            font_path,
            font_warning,
        )
        ext = "jpeg" if download_format == "JPEG" else "png"
        path_det = save_numpy_image_temp(annotated, ext, "detections")
        path_lineart = save_numpy_image_temp(lineart_export, ext, "lineart")
        if ctx and len(ctx["ft"]) > 0:
            hi = 0
            line_box, line_conf = lineart_views(ctx, ctx["ft"], hi)
            dd = gr.update(
                choices=[(f"[{i}] {ctx['ft'][i]}", str(i)) for i in range(len(ctx["ft"]))],
                value="0",
            )
        else:
            line_box, line_conf = lineart_export, lineart_preview
            dd = gr.update(choices=[], value=None)
        return (
            annotated,
            line_box,
            line_conf,
            summary,
            path_det,
            path_lineart,
            ctx,
            dd,
            "",
        )

    def _refresh_bbox_highlight(ctx, idx_value):
        if not ctx or idx_value is None:
            return gr.skip(), gr.skip()
        n = len(ctx["ft"])
        if n == 0:
            return gr.skip(), gr.skip()
        try:
            idx = int(str(idx_value))
        except (TypeError, ValueError):
            return gr.skip(), gr.skip()
        idx = max(0, min(idx, n - 1))
        return lineart_views(ctx, ctx["ft"], idx)

    def _replace_glyph(ctx, idx_value, new_glyph, download_format, ground_truth):
        if not ctx or idx_value is None:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Нет детекций.** Сначала загрузите изображение.",
            )
        n = len(ctx["ft"])
        if n == 0:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Нет знаков для замены.**",
            )
        try:
            idx = int(str(idx_value))
        except (TypeError, ValueError):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Некорректный индекс.**",
            )
        idx = max(0, min(idx, n - 1))
        repl = (new_glyph or "").strip()
        if not repl:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Введите новый символ** (поле ввода или список классов) — один или несколько юникод-знаков.",
            )
        new_ft = list(ctx["ft"])
        new_ft[idx] = repl
        new_ctx = {**ctx, "ft": new_ft}
        box, conf = lineart_views(new_ctx, new_ft, idx)
        clean = lineart_clean_only(new_ctx, new_ft)
        ext = "jpeg" if download_format == "JPEG" else "png"
        path = save_numpy_image_temp(clean, ext, "lineart")
        ordered = _reading_order_safe(new_ctx["fb"], new_ft, new_ctx["reading_alpha"])
        parts: list[str] = []
        if ordered:
            parts.append(f"**Строка после правки:** `{ordered}`")
        else:
            parts.append("**Строка после правки:** `(пусто)`")
        gt = (ground_truth or "").strip()
        if gt and ordered:
            parts.append(f"CER (после правок vs эталон): **{CER(ordered, gt):.4f}**")
        dd_upd = gr.update(
            choices=[(f"[{i}] {new_ft[i]}", str(i)) for i in range(len(new_ft))],
            value=str(idx),
        )
        return (
            new_ctx,
            box,
            conf,
            dd_upd,
            path,
            gr.update(value=None),
            "\n\n".join(parts),
        )

    def _sync_catalog_to_textbox(picked: str | None):
        if picked is None or str(picked).strip() == "":
            return gr.skip()
        return gr.update(value=str(picked).strip())

    def _resave_lineart_format(ctx, download_format):
        if not ctx or not ctx["ft"]:
            return gr.skip()
        clean = lineart_clean_only(ctx, ctx["ft"])
        ext = "jpeg" if download_format == "JPEG" else "png"
        return save_numpy_image_temp(clean, ext, "lineart")

    with gr.Blocks(title="Sumerian YOLO") as demo:
        gr.Markdown(
            "## Шумерская клинопись — детекция YOLO\n"
            "Загрузите изображение таблички. При необходимости введите эталонный текст для подсчёта CER."
        )
        with gr.Row():
            inp = gr.Image(label="Изображение", type="numpy", sources=["upload"])
            out_img = gr.Image(label="Детекции", type="numpy", format="png")
        with gr.Row():
            lineart_boxed = gr.Image(
                label="Автография: bbox знаков (оранжевая рамка — выбранный)",
                type="numpy",
                format="png",
                interactive=False,
            )
            lineart_conf_ref = gr.Image(
                label="То же + conf модели под знаком",
                type="numpy",
                format="png",
                interactive=False,
            )
        with gr.Row():
            glyph_select = gr.Dropdown(
                label="Выбор знака по bbox",
                choices=[],
                value=None,
            )
        with gr.Row():
            new_glyph_in = gr.Textbox(
                label="Новый символ (ввод вручную)",
                lines=1,
                max_lines=1,
                placeholder="Вставьте символ или выберите из списка справа",
            )
            glyph_catalog = gr.Dropdown(
                label="Или выберите из классов (dataset.yaml)",
                choices=catalog_pairs,
                value=None,
                filterable=True,
            )
            replace_glyph_btn = gr.Button("Заменить знак на автографии")
        with gr.Row():
            dl_fmt = gr.Radio(
                choices=["PNG", "JPEG"],
                value="PNG",
                label="Формат файла автографии (скачивание)",
            )
        with gr.Row():
            file_det = gr.File(label="Файл: детекции (.png / .jpg)")
            file_lineart = gr.File(
                label="Файл: автография (чистая, без рамок и conf)",
            )
        with gr.Row():
            conf_s = gr.Slider(0.05, 0.95, value=0.1, step=0.05, label="conf")
            iou_s = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="iou (NMS)")
            alpha_s = gr.Slider(0.1, 1.5, value=0.35, step=0.05, label="reading_order alpha")
        edited_lineart_md = gr.Markdown()
        gt_in = gr.Textbox(label="Эталонный текст (опционально, для CER)", lines=2)
        run_btn = gr.Button("Запустить")
        out_md = gr.Markdown()

        lineart_ctx = gr.State(value=None)

        predict_inputs = [inp, conf_s, iou_s, alpha_s, gt_in, dl_fmt]
        predict_outputs = [
            out_img,
            lineart_boxed,
            lineart_conf_ref,
            out_md,
            file_det,
            file_lineart,
            lineart_ctx,
            glyph_select,
            edited_lineart_md,
        ]

        run_btn.click(_predict, inputs=predict_inputs, outputs=predict_outputs)
        inp.change(_predict, inputs=predict_inputs, outputs=predict_outputs)
        for s in (conf_s, iou_s, alpha_s):
            s.release(_predict, inputs=predict_inputs, outputs=predict_outputs)
        gt_in.change(_predict, inputs=predict_inputs, outputs=predict_outputs)
        glyph_select.change(
            _refresh_bbox_highlight,
            inputs=[lineart_ctx, glyph_select],
            outputs=[lineart_boxed, lineart_conf_ref],
        )
        replace_glyph_btn.click(
            _replace_glyph,
            inputs=[lineart_ctx, glyph_select, new_glyph_in, dl_fmt, gt_in],
            outputs=[
                lineart_ctx,
                lineart_boxed,
                lineart_conf_ref,
                glyph_select,
                file_lineart,
                glyph_catalog,
                edited_lineart_md,
            ],
        )
        glyph_catalog.change(
            _sync_catalog_to_textbox,
            inputs=[glyph_catalog],
            outputs=[new_glyph_in],
        )
        dl_fmt.change(
            _resave_lineart_format,
            inputs=[lineart_ctx, dl_fmt],
            outputs=[file_lineart],
        )

    return demo


def main() -> None:
    default_model = (
        REPO_ROOT / "runs/detect/sumerian_yolo/unicode_topN_finalfinal-26/weights/best.pt"
    )
    default_data = REPO_ROOT / "dataset/yolo/dataset_unicode_topN/dataset.yaml"

    p = argparse.ArgumentParser(description="Gradio demo for Sumerian YOLO")
    p.add_argument("--model", type=Path, default=default_model, help="Path to best.pt")
    p.add_argument("--data", type=Path, default=default_data, help="Path to dataset.yaml")
    p.add_argument("--font", type=str, default=None, help="Path to Noto Sans Cuneiform font file")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    args = p.parse_args()

    if not args.model.is_file():
        sys.stderr.write(f"Model not found: {args.model}\n")
        sys.exit(1)
    if not args.data.is_file():
        sys.stderr.write(f"dataset.yaml not found: {args.data}\n")
        sys.exit(1)

    font_resolved, font_warn = resolve_font(args.font)
    demo = build_demo(args.model, args.data, font_resolved, font_warn)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
