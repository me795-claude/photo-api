"""Сборка тестового набора из документа «Требование к фото на пропуск.docx».

Шаги для каждой картинки-иллюстрации документа:
  1. найти отдельные фото (рамки «Не подходит / Подходит»);
  2. убрать значок ✘/✔ в правом нижнем углу, чтобы он не подсказывал модели ответ
     (угол заполняется зеркальным фрагментом левого нижнего угла — портрет почти симметричен);
  3. лёгкое шумоподавление артефактов JPEG;
  4. повысить разрешение нейросетью EDSR x4 (cv2.dnn_superres, веса EDSR_x4.pb);
  5. обрезать по центру до пропорций 35×45 и сохранить JPEG;
  6. записать эталонную разметку testset/expected.json.

Запуск:
  pip install -r tools/requirements-tools.txt
  curl -L -o EDSR_x4.pb https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb
  python tools/build_testset.py --docx "Требование к фото на пропуск.docx" --model EDSR_x4.pb --out testset
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ASPECT = 35.0 / 45.0

# Иллюстрации в порядке появления в документе: (требование, [(метка, слаг, описание, эталон)])
# Эталон: expect — пункты, результат которых обязан совпасть;
#         expect_failed_any — хотя бы один из пунктов должен быть failed.
S, F = "successed", "failed"
DOC = [
    ("format", [
        ("bad", "head_too_large", "Голова слишком крупная: лицо занимает почти весь кадр", {"expect": {"format": F}}),
        ("bad", "head_too_small", "Голова слишком мелкая: видны грудь и галстук, много фона", {"expect": {"format": F}}),
        ("good", "ok", "Голова занимает 32–36 мм из 45", {"expect": {"format": S}}),
    ]),
    ("quality", [
        ("bad", "blurry", "Снимок размыт", {"expect": {"quality": F}}),
        ("bad", "stains_folds", "Пятна и заломы на фото", {"expect": {"quality": F}}),
        ("good", "ok", "Чёткий снимок без дефектов", {"expect": {"quality": S}}),
    ]),
    ("lighting_color", [
        ("bad", "too_dark", "Затемнение: лицо слишком тёмное", {"expect": {"lighting_color": F}}),
        ("bad", "overexposed", "Пересвет: лицо выбелено", {"expect": {"lighting_color": F}}),
        ("good", "ok", "Равномерный свет, естественный цвет", {"expect": {"lighting_color": S}}),
    ]),
    ("background", [
        ("bad", "nature", "Фон — листва и трава, не однотонный", {"expect": {"background": F}}),
    ]),
    ("background", [
        ("good", "ok", "Однотонный светло-серый фон", {"expect": {"background": S}}),
    ]),
    ("headwear", [
        ("bad", "panama", "Панама на голове", {"expect": {"headwear": F}}),
        ("bad", "cap", "Кепка на голове", {"expect": {"headwear": F}}),
        ("good", "ok", "Без головного убора", {"expect": {"headwear": S}}),
    ]),
    ("head_position", [
        ("bad", "other_person", "В кадре виден другой человек, который держит ребёнка (пропорции исходные)",
         {"expect_failed_any": ["head_position", "background", "face_eyes"], "_align": "none"}),
        ("bad", "hands_hold_head", "Голову ребёнка поддерживают руками",
         {"expect_failed_any": ["head_position", "face_eyes"]}),
        ("good", "ok", "Голова прямо, анфас", {"expect": {"head_position": S}}),
    ]),
    ("gaze_expression", [
        ("bad", "looking_aside", "Взгляд в сторону от объектива", {"expect": {"gaze_expression": F}}),
    ]),
    ("gaze_expression", [
        ("good", "ok", "Взгляд в объектив, нейтральное выражение", {"expect": {"gaze_expression": S}}),
    ]),
    ("face_eyes", [
        ("bad", "hair_covers_face", "Волосы закрывают часть лица и глаз", {"expect": {"face_eyes": F}}),
        ("bad", "eyes_closed", "Глаза закрыты", {"expect": {"face_eyes": F}}),
        ("good", "ok", "Лицо открыто, глаза открыты", {"expect": {"face_eyes": S}}),
    ]),
    ("glasses", [
        ("bad", "dark_lenses", "Тёмные стёкла закрывают глаза", {"expect": {"glasses": F}}),
        ("bad", "glare", "Блики на стёклах очков", {"expect": {"glasses": F}}),
        ("good", "ok", "Тонкая оправа, прозрачные стёкла без бликов", {"expect": {"glasses": S}}),
    ]),
]


# ------------------------------------------------------------------ docx

def doc_images(docx: Path):
    """PNG-иллюстрации в порядке их появления в тексте документа."""
    with zipfile.ZipFile(docx) as z:
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
        rid2target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(media/[^"]+)"', rels))
        rid2target.update({k: v for v, k in re.findall(r'Target="(media/[^"]+)"[^>]*Id="(rId\d+)"', rels)})
        doc = z.read("word/document.xml").decode("utf-8")
        order = re.findall(r'r:embed="(rId\d+)"', doc)
        for rid in order:
            data = np.frombuffer(z.read("word/" + rid2target[rid]), np.uint8)
            yield rid2target[rid], cv2.imdecode(data, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ поиск фото

def _runs(mask: np.ndarray, min_len: int):
    """Непрерывные отрезки True длиной >= min_len."""
    runs, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def find_panels(img: np.ndarray, expected: int):
    """Фото в рамках: крупные связные области «не белых» пикселей.

    Красные/зелёные подписи и значки (насыщенные цвета) предварительно вырезаются,
    поэтому подписи под фото не сливаются с ним, а выступающая галочка не расширяет рамку.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue < 10) | (hue > 165)) & (sat > 90) & (val > 60)
    vivid_green = (hue > 50) & (hue < 80) & (sat > 150) & (val > 100)  # галочка и «Подходит», не листва
    saturated = (red | vivid_green).astype(np.uint8)
    saturated = cv2.dilate(saturated, np.ones((5, 5), np.uint8)) > 0
    content = ((img.min(axis=2) < 238) & ~saturated).astype(np.uint8)

    n, _, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh > 0.45 * h and bw > 0.12 * w:
            boxes.append((area, x, y, x + bw, y + bh))
    boxes = sorted(sorted(boxes, reverse=True)[:expected], key=lambda b: b[1])
    if len(boxes) != expected:
        raise RuntimeError(f"найдено {len(boxes)} фото вместо {expected}")

    panels = []
    for _, x0, y0, x1, y1 in boxes:
        # Уточняем рамку: ищем у каждого края (в пределах 40 px) линию, заполненную > 60 %.
        # Так отрезается выступающий за рамку хвост галочки, а белый фон внутри рамки не трогается.
        def edge(profile, from_end):
            idx = range(len(profile) - 1, max(len(profile) - 41, -1), -1) if from_end else range(min(41, len(profile)))
            for k in idx:
                if profile[k] > 0.6:
                    return k
            return len(profile) - 1 if from_end else 0
        box = content[y0:y1, x0:x1]
        cols, rows = box.mean(axis=0), box.mean(axis=1)
        x0, x1 = x0 + edge(cols, False), x0 + edge(cols, True) + 1
        y0, y1 = y0 + edge(rows, False), y0 + edge(rows, True) + 1
        inset = 4  # убрать тонкую серую рамку
        panels.append((x0 + inset, y0 + inset, x1 - inset, y1 - inset))
    return panels


# ------------------------------------------------------------------ удаление значка

def remove_mark(photo: np.ndarray) -> np.ndarray:
    h, w = photo.shape[:2]
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue < 8) | (hue > 170)) & (sat > 120) & (val > 120)
    green = (hue > 38) & (hue < 88) & (sat > 70) & (val > 50)
    corner = np.zeros_like(red)
    corner[int(h * 0.55):, int(w * 0.55):] = True
    red &= corner
    green &= corner
    mark = red if red.sum() >= 30 else green  # крестик красный, галочка зелёная; листва фона не в счёт
    ys, xs = np.nonzero(mark)
    if len(xs) < 30:
        return photo
    pad = int(0.05 * w)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    mask = np.zeros((h, w), np.float32)
    mask[y0:, x0:] = 1.0  # значок стоит в углу — закрываем до краёв кадра
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad / 2.5)
    mask = np.clip(mask * 1.6, 0, 1)[..., None]
    mirrored = photo[:, ::-1]
    out = photo.astype(np.float32) * (1 - mask) + mirrored.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ апскейл

class EDSR:
    """EDSR x4 через cv2.dnn_superres (нужен opencv-contrib: слой DepthToSpace регистрирует именно он)."""

    def __init__(self, model_path: Path):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(str(model_path))
        self.sr.setModel("edsr", 4)

    def upscale(self, img: np.ndarray) -> np.ndarray:
        return self.sr.upsample(img)


def crop_aspect(img: np.ndarray, align: str = "center") -> np.ndarray:
    if align == "none":  # оставить исходные пропорции (иначе обрезка уберёт сам дефект)
        return img
    h, w = img.shape[:2]
    if w / h > ASPECT:  # слишком широкое — режем по бокам
        nw = int(round(h * ASPECT))
        x0 = {"left": 0, "right": w - nw}.get(align, (w - nw) // 2)
        return img[:, x0: x0 + nw]
    nh = int(round(w / ASPECT))  # слишком высокое — режем снизу (голову не трогаем)
    return img[:nh]


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path, help="EDSR_x4.pb")
    ap.add_argument("--out", default=Path("testset"), type=Path)
    ap.add_argument("--debug", type=Path, help="папка для промежуточных картинок и контактного листа")
    args = ap.parse_args()

    images = list(doc_images(args.docx))
    if len(images) != len(DOC):
        print(f"в документе {len(images)} иллюстраций, ожидалось {len(DOC)}", file=sys.stderr)
        return 1

    out_img = args.out / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    if args.debug:
        args.debug.mkdir(parents=True, exist_ok=True)
    sr = EDSR(args.model)

    cases, n = [], 0
    thumbs = []
    for (src_name, img), (req, items) in zip(images, DOC):
        panels = find_panels(img, len(items))
        for (x0, y0, x1, y1), (label, slug, desc, exp) in zip(panels, items):
            n += 1
            case_id = f"{n:02d}_{req}_{label}_{slug}"
            photo = img[y0:y1, x0:x1].copy()
            clean = remove_mark(photo)
            clean = cv2.fastNlMeansDenoisingColored(clean, None, 3, 3, 7, 21)
            t = time.time()
            big = sr.upscale(clean)
            big = crop_aspect(big, exp.get("_align", "center"))
            path = out_img / f"{case_id}.jpg"
            cv2.imwrite(str(path), big, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{case_id}: {photo.shape[1]}×{photo.shape[0]} → {big.shape[1]}×{big.shape[0]}"
                  f"  ({time.time() - t:.1f} с)  из {src_name}")
            if args.debug:
                cv2.imwrite(str(args.debug / f"{case_id}_src.png"), photo)
                cv2.imwrite(str(args.debug / f"{case_id}_clean.png"), clean)
                thumbs.append((case_id, cv2.resize(big, (210, 270), interpolation=cv2.INTER_AREA)))
            case = {
                "id": case_id,
                "file": f"images/{case_id}.jpg",
                "requirement": req,
                "label": label,
                "description": desc,
                "source": f"{args.docx.name}: {src_name}",
                "expect_overall": "successed" if label == "good" else "failed",
            }
            case.update({k: v for k, v in exp.items() if not k.startswith("_")})
            cases.append(case)

    expected = {
        "description": "Тестовый набор из документа «Требование к фото на пропуск». "
                       "Фото нарезаны из иллюстраций документа, значки ✘/✔ удалены, разрешение повышено EDSR x4.",
        "rules": "expect — пункты, результат которых должен совпасть; expect_failed_any — хотя бы один failed; "
                 "expect_overall — итоговый вердикт (good → successed, bad → failed).",
        "cases": cases,
    }
    (args.out / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.debug and thumbs:
        cols = 7
        rows = (len(thumbs) + cols - 1) // cols
        sheet = np.full((rows * 300, cols * 220, 3), 255, np.uint8)
        for i, (cid, th) in enumerate(thumbs):
            r, c = divmod(i, cols)
            sheet[r * 300 + 5: r * 300 + 275, c * 220 + 5: c * 220 + 215] = th
            cv2.putText(sheet, cid[:2] + " " + cid.split("_", 2)[2][:22], (c * 220 + 5, r * 300 + 292),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.debug / "contact_sheet.jpg"), sheet)

    print(f"\nГотово: {len(cases)} фото → {out_img}, эталон → {args.out / 'expected.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
