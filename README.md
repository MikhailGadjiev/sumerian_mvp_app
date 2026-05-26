## Repository structure

```text
modules/
├── ocr/
├── detection/
├── classification/
└── translation/# Sumerian Translator MVP

This repository contains the MVP application layer for the Sumerian cuneiform recognition and translation system.

The application is implemented with Gradio and integrates the main stages of the proposed pipeline:

Image → OCR / sign detection → Transliteration → Machine translation → Manual correction

## Research modules

This repository also contains lightweight copies of the research module code used in the project:

- `modules/ocr` — OCR and ATF conversion scripts
- `modules/detection` — sign detection code based on YOLO and Faster R-CNN experiments
- `modules/classification` — sign classification training and evaluation code with experiment configs
- `modules/translation` — machine translation training, inference and metric scripts

Large model checkpoints, datasets and training artifacts are not included in this repository.


## Purpose

The goal of this repository is to demonstrate the user-facing MVP interface and the modular integration of the research components developed in separate repositories.

## Main features

- Gradio-based user interface
- Image upload for cuneiform tablets
- OCR pipeline integration
- Transliteration output
- Translation module integration
- Human-in-the-loop manual correction
- Modular architecture

## Project structure

- `app.py` — Gradio MVP application
- `requirements.txt` — Python dependencies
- `README.md` — project description

## Authors and contribution

- Mikhail Gadiev — MVP architecture, Gradio interface, integration logic, user scenario, expert workflow
- Maxim Partin — ML models, OCR, sign classification/detection and translation experiments

## Notes

The trained model checkpoints are not included in this repository because of their large size.

The application supports connecting external model weights through environment variables:

```bash
export YOLO_MODEL=/path/to/best.pt
export TRANSLATION_MODEL_DIR=/path/to/saved/byt5_model
python3 app.py 
