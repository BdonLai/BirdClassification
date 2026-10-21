COS30082 – Bird Species Classifier (Streamlit)

Interactive demo for 200-class bird species classification using six DenseNet-121 model variants trained for Assignment 1.

Features

Mode 1 – Single model
Upload one image
Choose any of the six models
Predict → Top-5 classes with probabilities
Bar chart + pie chart visualisation


Mode 2 – Compare all models
Upload one image
Predict with all six models
Table of each model’s Top-1 prediction + confidence
Comparison bar chart + majority-vote summary

Models
Name
Weight file
Role
Baseline (Full)
densenet121_baseline_best.pth
Full fine-tune baseline + strong augmentation
Fine-tuned (Full)
densenet121_finetuned_best.pth
Further fine-tuned full model
Baseline (No Aug)
densenet121_baseline_no_aug_best.pth
Ablation: no data augmentation
Fine-tuned (No Aug)
densenet121_finetuned_no_aug_best.pth
Fine-tune of no-aug baseline
Baseline (Frozen)
densenet121_baseline_frozen_best.pth
Ablation: frozen backbone
Fine-tuned (Unfrozen)
densenet121_finetuned_unfrozen_best.pth
Unfreeze + fine-tune from frozen baseline
Architecture (all six): DenseNet-121 + Dropout(0.5) + Linear(1024 → 200).

Repository layout

.
├── app.py
├── requirements.txt
├── README.md
├── classes.txt          # optional: 200 lines, one class name per line
└── weights/
    ├── densenet121_baseline_best.pth
    ├── densenet121_finetuned_best.pth
    ├── densenet121_baseline_no_aug_best.pth
    ├── densenet121_finetuned_no_aug_best.pth
    ├── densenet121_baseline_frozen_best.pth
    └── densenet121_finetuned_unfrozen_best.pth

If classes.txt is missing, the app labels classes as Class_000 … Class_199.

Local run

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Open the URL shown in the terminal (usually http://localhost:8501).

Deploy on Streamlit Community Cloud





Push this repo to GitHub (including the weights/ folder, or use Git LFS if files are large).



Go to share.streamlit.io → New app.



Select the repository, branch, and main file: app.py.



Deploy.

Notes





Free Cloud instances use CPU only; first prediction may be slow while models load (they are cached afterward).



GitHub has a soft limit on large files; if a .pth exceeds ~100 MB, use Git LFS or host weights elsewhere and download at startup.



Ensure requirements.txt is at the repo root.

Inference details





Input: RGB image (jpg/png/webp/bmp)



Preprocess: resize shorter side logic via Resize(258) → CenterCrop(224) → ImageNet mean/std normalisation



Output: softmax probabilities over 200 classes

This matches the evaluation pipeline used in the training notebooks.

Assignment context

Trained models report Top-1 accuracy and average accuracy per class, plus extended test metrics (Top-5, macro F1, precision, recall) in the notebooks. This app is for interactive demonstration and qualitative comparison of the six training strategies.

License

Academic use – COS30082 Assignment 1.