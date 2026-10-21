"""
COS30082 – Bird Species Classification
Streamlit demo app for 6 DenseNet-121 model variants.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bird Species Classifier | COS30082",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 200
IMAGE_SIZE = 224
DROPOUT = 0.5
WEIGHTS_DIR = Path(__file__).parent / "Weights"

MODEL_REGISTRY = {
    "Baseline (Full)": {
        "file": "densenet121_baseline_best.pth",
        "description": (
            "Standard transfer-learning baseline. DenseNet-121 pretrained on ImageNet, "
            "all layers trainable, strong data augmentation + MixUp. "
            "Good generalisation starting point."
        ),
    },
    "Fine-tuned (Full)": {
        "file": "densenet121_finetuned_best.pth",
        "description": (
            "Full model fine-tuned from the best baseline with a lower learning rate, "
            "stronger MixUp, and early stopping. Usually the strongest overall performer."
        ),
    },
    "Baseline (No Aug)": {
        "file": "densenet121_baseline_no_aug_best.pth",
        "description": (
            "Ablation: same architecture as the full baseline but **without** data "
            "augmentation or MixUp. Expected to overfit more; useful for comparison."
        ),
    },
    "Fine-tuned (No Aug)": {
        "file": "densenet121_finetuned_no_aug_best.pth",
        "description": (
            "Fine-tuned continuation of the no-augmentation baseline. "
            "Shows how much fine-tuning helps when regularisation from aug is missing."
        ),
    },
    "Baseline (Frozen)": {
        "file": "densenet121_baseline_frozen_best.pth",
        "description": (
            "Ablation: ImageNet backbone **frozen**; only the classifier head is trained. "
            "Faster to train, lower capacity to adapt to fine-grained bird features."
        ),
    },
    "Fine-tuned (Unfrozen)": {
        "file": "densenet121_finetuned_unfrozen_best.pth",
        "description": (
            "Starts from the frozen baseline, then **unfreezes all layers** and fine-tunes "
            "with a small learning rate. Demonstrates the gain from full adaptation."
        ),
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model definition (must match training notebooks)
# ---------------------------------------------------------------------------
def create_densenet(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Class names (optional classes.txt: one name per line, 200 lines)
# ---------------------------------------------------------------------------
def load_class_names() -> list[str]:
    candidates = [
        Path(__file__).parent / "classes.txt",
        WEIGHTS_DIR / "classes.txt",
        Path(__file__).parent / "data" / "classes.txt",
    ]
    for path in candidates:
        if path.exists():
            names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(names) >= NUM_CLASSES:
                return names[:NUM_CLASSES]
            # pad if short
            while len(names) < NUM_CLASSES:
                names.append(f"Class_{len(names):03d}")
            return names
    return [f"Class_{i:03d}" for i in range(NUM_CLASSES)]


CLASS_NAMES = load_class_names()


# ---------------------------------------------------------------------------
# Caching: load each model once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    info = MODEL_REGISTRY[model_key]
    weight_path = WEIGHTS_DIR / info["file"]
    if not weight_path.exists():
        return None, f"Weight file not found: `{weight_path}`"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_densenet()
    try:
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    except Exception as e:
        return None, f"Failed to load `{info['file']}`: {e}"

    model.to(device)
    model.eval()
    meta = {}
    if isinstance(ckpt, dict):
        meta = {
            "epoch": ckpt.get("epoch"),
            "val_top1": ckpt.get("val_top1"),
            "val_avg_per_class": ckpt.get("val_avg_per_class"),
        }
    return (model, device, meta), None


def predict_topk(model, device, image: Image.Image, k: int = 5):
    preprocess = get_preprocess()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i]), CLASS_NAMES[i]) for i in top_idx]


def predict_top1(model, device, image: Image.Image):
    results = predict_topk(model, device, image, k=1)
    return results[0]


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_topk_bar(topk_results, title: str):
    labels = [f"{name}\n(id {idx})" for idx, _, name in topk_results]
    values = [p * 100 for _, p, _ in topk_results]
    colors = plt.cm.Blues(np.linspace(0.85, 0.35, len(values)))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(range(len(values))[::-1], values, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(values))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_xlim(0, max(100, max(values) * 1.15))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for bar, v in zip(bars, values):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_topk_pie(topk_results, title: str):
    labels = [f"{name}" for _, _, name in topk_results]
    values = [p for _, p, _ in topk_results]
    others = max(0.0, 1.0 - sum(values))
    if others > 1e-4:
        labels = labels + ["Other classes"]
        values = values + [others]

    colors = list(plt.cm.Set3.colors[: len(values)])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=90,
        colors=colors,
        pctdistance=0.72,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    ax.legend(
        wedges,
        [f"{l} ({v*100:.1f}%)" for l, v in zip(labels, values)],
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_all_models_bar(rows: list[dict]):
    """rows: list of {model, class_id, class_name, prob}"""
    names = [r["model"] for r in rows]
    probs = [r["prob"] * 100 for r in rows]
    pred_labels = [f"{r['class_name']} ({r['prob']*100:.1f}%)" for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(names)))
    bars = ax.barh(y_pos, probs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Top-1 Confidence (%)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_title("Top-1 Prediction Confidence — All Models", fontsize=12, fontweight="bold")
    for bar, lab in zip(bars, pred_labels):
        ax.text(
            min(bar.get_width() + 1.5, 90),
            bar.get_y() + bar.get_height() / 2,
            lab,
            va="center",
            fontsize=8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def sidebar_info():
    st.sidebar.title("🐦 Bird Species Classifier")
    st.sidebar.markdown(
        """
**COS30082 Assignment 1**  
DenseNet-121 · 200 bird species

This app runs **six trained model variants** so you can compare
transfer learning, data augmentation, and freezing strategies.
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Available models")
    for name, info in MODEL_REGISTRY.items():
        path = WEIGHTS_DIR / info["file"]
        status = "✅" if path.exists() else "❌"
        st.sidebar.markdown(f"{status} **{name}**")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Upload a clear photo of a bird. "
        "Models expect a single bird roughly centred in the frame."
    )
    device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.caption(f"Inference device: **{device}**")


def show_model_card(model_key: str, meta: dict | None = None):
    info = MODEL_REGISTRY[model_key]
    st.markdown(f"### {model_key}")
    st.info(info["description"])
    if meta:
        cols = st.columns(3)
        if meta.get("epoch") is not None:
            cols[0].metric("Checkpoint epoch", meta["epoch"])
        if meta.get("val_top1") is not None:
            cols[1].metric("Val Top-1 (train-time)", f"{meta['val_top1']*100:.2f}%")
        if meta.get("val_avg_per_class") is not None:
            cols[2].metric("Val Avg/Class", f"{meta['val_avg_per_class']*100:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sidebar_info()

    st.title("Bird Species Classification")
    st.markdown(
        """
Use this demo to **upload a bird image** and run prediction with one or all of the six
DenseNet-121 models trained for COS30082 (CUB-style 200-class dataset).

| Mode | What it does |
|------|----------------|
| **Single model** | Choose one model → Top-5 classes + bar chart + pie chart |
| **Compare all models** | Run all 6 models → each model's Top-1 + comparison bar chart |
"""
    )

    mode = st.radio(
        "Prediction mode",
        ["Single model (Top-5 + charts)", "Compare all models (Top-1 each)"],
        horizontal=True,
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload a bird image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="RGB image recommended. The app resizes and centre-crops to 224×224.",
    )

    if uploaded is None:
        st.markdown(
            """
### How to use
1. Upload an image using the control above.  
2. Choose a mode.  
3. Click **Predict**.  

**Tips for better results**
- Prefer a single bird, not a flock  
- Avoid heavy occlusion and extreme blur  
- Natural lighting helps  

**About the metrics you may have seen in training**  
Models were selected using **Top-1 accuracy** and **average accuracy per class** on a
held-out validation split, then evaluated on the official test set (Top-1, Top-5,
macro F1, etc.).
"""
        )
        return

    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return

    col_img, col_meta = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_meta:
        st.markdown("**Image info**")
        st.write(f"- Filename: `{uploaded.name}`")
        st.write(f"- Size: {image.size[0]} × {image.size[1]} px")
        st.write(f"- Mode: {image.mode}")
        st.markdown(
            "Preprocessing: resize → centre crop **224×224** → "
            "ImageNet normalisation (same as training/eval notebooks)."
        )

    st.markdown("---")

    # -------------------- Mode A: single model --------------------
    if mode.startswith("Single"):
        model_key = st.selectbox(
            "Select model",
            list(MODEL_REGISTRY.keys()),
            help="Each option loads a different trained weight file from the weights/ folder.",
        )
        show_model_card(model_key)

        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner(f"Loading **{model_key}** and running inference..."):
                loaded, err = load_model(model_key)
                if err:
                    st.error(err)
                    st.stop()
                model, device, meta = loaded
                topk = predict_topk(model, device, image, k=5)

            st.success("Prediction complete")
            if meta:
                show_model_card(model_key, meta)

            st.subheader("Top-5 predictions")
            df = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Class ID": idx,
                        "Class name": name,
                        "Probability": f"{prob*100:.2f}%",
                        "Confidence": prob,
                    }
                    for i, (idx, prob, name) in enumerate(topk)
                ]
            )
            st.dataframe(
                df[["Rank", "Class ID", "Class name", "Probability"]],
                use_container_width=True,
                hide_index=True,
            )

            # Highlight top-1
            top_idx, top_p, top_name = topk[0]
            st.markdown(
                f"### Top prediction: **{top_name}** (class {top_idx}) — "
                f"**{top_p*100:.2f}%** confidence"
            )

            c1, c2 = st.columns(2)
            with c1:
                fig_bar = plot_topk_bar(topk, f"Top-5 — {model_key}")
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)
            with c2:
                fig_pie = plot_topk_pie(topk, f"Probability mass — {model_key}")
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)

            st.caption(
                "Bar chart: probability of each of the top-5 classes. "
                "Pie chart: share of probability mass among top-5 vs remaining classes."
            )

    # -------------------- Mode B: all models --------------------
    else:
        st.markdown(
            """
### Compare all models
Each model outputs its **Top-1** class and confidence for the same image.
Use this to see agreement / disagreement across training strategies.
"""
        )
        if st.button("Predict with all 6 models", type="primary", use_container_width=True):
            rows = []
            progress = st.progress(0.0, text="Running models...")
            keys = list(MODEL_REGISTRY.keys())
            for i, model_key in enumerate(keys):
                progress.progress((i) / len(keys), text=f"Running: {model_key}")
                loaded, err = load_model(model_key)
                if err:
                    rows.append(
                        {
                            "model": model_key,
                            "class_id": None,
                            "class_name": f"Error: {err}",
                            "prob": 0.0,
                            "ok": False,
                        }
                    )
                    continue
                model, device, meta = loaded
                idx, prob, name = predict_top1(model, device, image)
                rows.append(
                    {
                        "model": model_key,
                        "class_id": idx,
                        "class_name": name,
                        "prob": prob,
                        "ok": True,
                        "meta": meta,
                    }
                )
            progress.progress(1.0, text="Done")
            progress.empty()

            st.success("All available models finished")

            table = pd.DataFrame(
                [
                    {
                        "Model": r["model"],
                        "Top-1 Class ID": r["class_id"] if r["ok"] else "—",
                        "Class name": r["class_name"],
                        "Confidence": f"{r['prob']*100:.2f}%" if r["ok"] else "—",
                    }
                    for r in rows
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)

            ok_rows = [r for r in rows if r["ok"]]
            if ok_rows:
                # Agreement summary
                from collections import Counter

                votes = Counter(r["class_id"] for r in ok_rows)
                majority_id, majority_n = votes.most_common(1)[0]
                st.markdown(
                    f"**Majority vote:** class **{majority_id}** "
                    f"({CLASS_NAMES[majority_id]}) — "
                    f"{majority_n}/{len(ok_rows)} models agree."
                )

                fig = plot_all_models_bar(ok_rows)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Horizontal bars show each model's Top-1 confidence. "
                    "Labels on the bars are the predicted class name and probability."
                )

            with st.expander("Model descriptions"):
                for key in keys:
                    st.markdown(f"**{key}** — {MODEL_REGISTRY[key]['description']}")


if __name__ == "__main__":
    main()