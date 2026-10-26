# =============================================================================
# TerraVision AI — Satellite Image Classification
# Streamlit + PyTorch (ResNet18 fine-tuned on EuroSAT RGB)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Imports and configuration
# -----------------------------------------------------------------------------
from html import escape
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="TerraVision AI | Satellite Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 2. Model configuration and class metadata
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_resnet18_finetuned.pth"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_DETAILS = {
    "AnnualCrop": ("Agricultural land", "Cultivated fields used for seasonal crops."),
    "Forest": ("Forest cover", "Dense tree canopy and forest vegetation."),
    "HerbaceousVegetation": ("Natural vegetation", "Grassland and herbaceous plant cover."),
    "Highway": ("Transport infrastructure", "Road networks and major highways."),
    "Industrial": ("Industrial land", "Industrial facilities and built-up areas."),
    "Pasture": ("Pasture land", "Open land used for grazing."),
    "PermanentCrop": ("Permanent agriculture", "Orchards, vineyards, and perennial crops."),
    "Residential": ("Residential area", "Housing and developed urban neighborhoods."),
    "River": ("River system", "River channels and flowing inland water."),
    "SeaLake": ("Water body", "Lakes, seas, and other large water surfaces."),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Presentation-only metadata (does not affect the model or its class order).
CLASS_LABELS = {
    "AnnualCrop": "Annual Crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation",
    "Highway": "Highway",
    "Industrial": "Industrial",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent Crop",
    "Residential": "Residential",
    "River": "River",
    "SeaLake": "Sea or Lake",
}

# Minimal line icons (24x24 grid, stroke-based). Inline SVG: no extra dependency.
CLASS_ICONS = {
    "AnnualCrop": '<path d="M3 7h18M3 12h18M3 17h18"/>',
    "Forest": '<path d="M12 3l6 8h-3l4 6H5l4-6H6z"/><path d="M12 17v4"/>',
    "HerbaceousVegetation": (
        '<path d="M12 21V11"/><path d="M12 15c-3.5 0-5.5-2-5.5-5.5 3.5 0 5.5 2 5.5 5.5z"/>'
        '<path d="M12 12.5c3-.2 5-2 5-5-3 .2-5 2-5 5z"/>'
    ),
    "Highway": '<path d="M8 3L5 21"/><path d="M16 3l3 18"/><path d="M12 4v3M12 10.5v3M12 17v3"/>',
    "Industrial": '<path d="M3 21V10l6 4v-4l6 4V5h3v16H3z"/>',
    "Pasture": '<path d="M2 18c3-6 6-6 9 0 2.5-3.5 5.5-3.5 11 0"/><path d="M2 21h20"/>',
    "PermanentCrop": '<circle cx="12" cy="9" r="5"/><path d="M12 14v7M9 21h6"/>',
    "Residential": '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "River": '<path d="M3 8c3-3 6 3 9 0s6 3 9 0"/><path d="M3 15c3-3 6 3 9 0s6 3 9 0"/>',
    "SeaLake": '<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>',
}

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "jfif"]
LOW_CONFIDENCE_THRESHOLD = 60.0  # Only used to show a gentle, optional note.


# -----------------------------------------------------------------------------
# 3. Model loading and prediction functions (inference logic unchanged)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Ensure the models folder contains "
            "best_resnet18_finetuned.pth."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image, model):
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(probabilities, k=3)

    return [
        {
            "class": CLASS_NAMES[index.item()],
            "confidence": value.item() * 100,
        }
        for value, index in zip(values, indices)
    ]


# -----------------------------------------------------------------------------
# UI helpers (presentation only)
# -----------------------------------------------------------------------------
def render_html(markup: str) -> None:
    """Render an HTML snippet. Lines are stripped and joined so that Markdown
    never mistakes indented markup for a code block."""
    compact = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


def icon_svg(paths: str, size: int = 20) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


BRAND_ICON = (
    '<svg viewBox="0 0 32 32" width="26" height="26" fill="none" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="16" cy="16" r="8"/>'
    '<path d="M8 16h16"/>'
    '<path d="M16 8c-3.2 3.4-3.2 12.6 0 16"/>'
    '<path d="M16 8c3.2 3.4 3.2 12.6 0 16"/>'
    '<ellipse cx="16" cy="16" rx="13.5" ry="5" transform="rotate(-28 16 16)" opacity="0.55"/>'
    '<circle cx="27" cy="9.2" r="1.7" fill="currentColor" stroke="none"/>'
    '</svg>'
)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def bar_width(value: float) -> str:
    return f"{min(100.0, max(0.0, value)):.2f}"


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def section_header(title: str, copy: str = "", anchor: str = "") -> None:
    anchor_attr = f' id="{anchor}"' if anchor else ""
    copy_html = f'<p class="tv-section-copy">{copy}</p>' if copy else ""
    render_html(
        f'<div class="tv-section"{anchor_attr}>'
        f'<h2 class="tv-section-title">{title}</h2>{copy_html}</div>'
    )


# -----------------------------------------------------------------------------
# 4. Custom styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --tv-bg: #0B1120;
    --tv-surface: #111827;
    --tv-surface-2: #182335;
    --tv-accent: #42D6B0;
    --tv-accent-soft: rgba(66, 214, 176, 0.10);
    --tv-accent-line: rgba(66, 214, 176, 0.35);
    --tv-forest: #1F4D3F;
    --tv-text: #F3F6FA;
    --tv-muted: #9AA9BC;
    --tv-border: #273449;
    --tv-radius: 16px;
    --tv-radius-sm: 10px;
    --tv-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* ---- App shell ---------------------------------------------------------- */
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, rgba(31, 77, 63, 0.28), transparent 38%),
        var(--tv-bg);
    color: var(--tv-text);
    font-family: var(--tv-font);
}

[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1160px;
    padding: 2rem 1.5rem 3rem;
}

.tv-header, .tv-hero, .tv-section, .tv-card, .tv-upload-intro, .tv-notice,
.tv-footer, .tv-grid, .tv-meta, .tv-rank {
    font-family: var(--tv-font);
}

/* ---- Header ------------------------------------------------------------- */
.tv-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--tv-border);
}

.tv-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.tv-logo {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--tv-accent);
    background: var(--tv-surface-2);
    border: 1px solid var(--tv-border);
    border-radius: 12px;
}

.tv-brand-name {
    color: var(--tv-text);
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -0.2px;
    line-height: 1.2;
}

.tv-brand-sub {
    color: var(--tv-muted);
    font-size: 12px;
    line-height: 1.3;
    margin-top: 2px;
}

.tv-nav {
    display: flex;
    gap: 4px;
}

.tv-nav a {
    color: var(--tv-muted) !important;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 500;
    padding: 7px 12px;
    border-radius: 8px;
}

.tv-nav a:hover {
    color: var(--tv-text) !important;
    background: var(--tv-surface-2);
}

.tv-nav a:focus-visible {
    outline: 2px solid var(--tv-accent);
    outline-offset: 2px;
}

.tv-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--tv-muted);
    font-size: 12.5px;
    background: var(--tv-surface);
    border: 1px solid var(--tv-border);
    border-radius: 999px;
    padding: 6px 12px;
}

.tv-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--tv-accent);
    display: inline-block;
    flex: none;
}

.tv-dot.warn {
    background: #E5A94B;
}

/* ---- Hero --------------------------------------------------------------- */
.tv-hero {
    padding: 44px 0 8px;
    max-width: 760px;
}

.tv-badge {
    display: inline-block;
    color: var(--tv-accent);
    background: var(--tv-accent-soft);
    border: 1px solid var(--tv-accent-line);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 1.2px;
    margin-bottom: 18px;
}

.tv-hero-title {
    color: var(--tv-text);
    font-size: clamp(30px, 4.2vw, 44px);
    font-weight: 700;
    letter-spacing: -1.2px;
    line-height: 1.15;
    margin: 0 0 14px;
}

.tv-hero-copy {
    color: var(--tv-muted);
    font-size: 16.5px;
    line-height: 1.65;
    margin: 0 0 24px;
    max-width: 620px;
}

.tv-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.tv-chip {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: var(--tv-surface);
    border: 1px solid var(--tv-border);
    border-radius: var(--tv-radius-sm);
    padding: 9px 14px;
}

.tv-chip svg {
    color: var(--tv-accent);
    flex: none;
}

.tv-chip-value {
    color: var(--tv-text);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.2;
}

.tv-chip-label {
    color: var(--tv-muted);
    font-size: 12px;
    line-height: 1.2;
}

/* ---- Section headings --------------------------------------------------- */
.tv-section {
    margin: 40px 0 16px;
    padding-top: 28px;
    border-top: 1px solid var(--tv-border);
    scroll-margin-top: 24px;
}

.tv-section-title {
    color: var(--tv-text) !important;
    font-family: var(--tv-font) !important;
    font-size: 21px !important;
    font-weight: 650 !important;
    letter-spacing: -0.4px;
    line-height: 1.3 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.tv-section-copy {
    color: var(--tv-muted);
    font-size: 14.5px;
    line-height: 1.6;
    margin: 6px 0 0;
    max-width: 680px;
}

/* ---- Cards -------------------------------------------------------------- */
.tv-card {
    background: var(--tv-surface);
    border: 1px solid var(--tv-border);
    border-radius: var(--tv-radius);
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
}

.tv-label {
    color: var(--tv-muted);
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 10px;
}

/* ---- Upload ------------------------------------------------------------- */
.tv-upload-intro {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 12px;
}

.tv-upload-icon {
    width: 40px;
    height: 40px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--tv-accent);
    background: var(--tv-accent-soft);
    border: 1px solid var(--tv-accent-line);
    border-radius: 11px;
}

.tv-upload-title {
    color: var(--tv-text);
    font-size: 15.5px;
    font-weight: 600;
    line-height: 1.3;
}

.tv-upload-copy {
    color: var(--tv-muted);
    font-size: 13.5px;
    line-height: 1.55;
    margin-top: 3px;
}

.tv-formats {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.tv-format {
    color: var(--tv-muted);
    background: var(--tv-surface-2);
    border: 1px solid var(--tv-border);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 500;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--tv-surface);
    border: 1.5px dashed #3A4D68;
    border-radius: var(--tv-radius);
    padding: 26px 20px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--tv-accent);
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--tv-surface-2);
    color: var(--tv-text);
    border: 1px solid var(--tv-border);
    border-radius: 8px;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--tv-accent);
    color: var(--tv-accent);
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: var(--tv-muted);
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    color: var(--tv-text);
}

.tv-hint {
    color: var(--tv-muted);
    font-size: 12.5px;
    margin: 4px 0 0;
}

.tv-notice {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    color: var(--tv-muted);
    background: var(--tv-accent-soft);
    border: 1px solid var(--tv-accent-line);
    border-radius: var(--tv-radius-sm);
    padding: 10px 14px;
    font-size: 13.5px;
    margin: 14px 0 4px;
}

.tv-notice b {
    color: var(--tv-text);
    font-weight: 600;
    overflow-wrap: anywhere;
}

/* ---- Empty state -------------------------------------------------------- */
.tv-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 18px;
    margin-top: 16px;
}

.tv-step-title {
    color: var(--tv-text);
    font-size: 14.5px;
    font-weight: 600;
}

.tv-step-copy {
    color: var(--tv-muted);
    font-size: 13.5px;
    line-height: 1.55;
    margin-top: 3px;
}

/* ---- Category / info grids --------------------------------------------- */
.tv-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 12px;
}

.tv-grid.wide {
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.tv-cat {
    background: var(--tv-surface);
    border: 1px solid var(--tv-border);
    border-radius: 14px;
    padding: 16px;
}

.tv-cat-icon {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--tv-accent);
    background: var(--tv-accent-soft);
    border-radius: 9px;
    margin-bottom: 12px;
}

.tv-cat-name {
    color: var(--tv-text);
    font-size: 14.5px;
    font-weight: 600;
    line-height: 1.3;
}

.tv-cat-copy {
    color: var(--tv-muted);
    font-size: 13px;
    line-height: 1.5;
    margin-top: 4px;
}

/* ---- Results ------------------------------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--tv-surface);
    border-color: var(--tv-border);
    border-radius: var(--tv-radius);
}

[data-testid="stImage"] img {
    border-radius: 12px;
}

.tv-meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--tv-border);
}

.tv-meta span {
    display: block;
    color: var(--tv-muted);
    font-size: 12px;
}

.tv-meta b {
    display: block;
    color: var(--tv-text);
    font-size: 13.5px;
    font-weight: 500;
    margin-top: 2px;
    overflow-wrap: anywhere;
}

.tv-result-name {
    color: var(--tv-text);
    font-size: clamp(24px, 3vw, 30px);
    font-weight: 700;
    letter-spacing: -0.6px;
    line-height: 1.2;
    margin: 0 0 10px;
    overflow-wrap: anywhere;
}

.tv-tag {
    display: inline-block;
    color: var(--tv-accent);
    background: var(--tv-accent-soft);
    border: 1px solid var(--tv-accent-line);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12.5px;
    font-weight: 500;
    margin-bottom: 10px;
}

.tv-result-desc {
    color: var(--tv-muted);
    font-size: 14.5px;
    line-height: 1.6;
    margin: 0 0 22px;
}

.tv-confidence {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.tv-confidence-value {
    color: var(--tv-accent);
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1;
}

.tv-confidence-caption {
    color: var(--tv-muted);
    font-size: 13px;
}

.tv-bar {
    width: 100%;
    height: 8px;
    background: var(--tv-surface-2);
    border-radius: 999px;
    overflow: hidden;
}

.tv-bar-fill {
    height: 100%;
    background: var(--tv-accent);
    border-radius: 999px;
}

.tv-bar-fill.muted {
    background: #55667F;
}

.tv-note {
    color: var(--tv-muted);
    font-size: 13px;
    line-height: 1.55;
    margin-top: 16px;
    padding: 10px 12px;
    background: var(--tv-surface-2);
    border-radius: var(--tv-radius-sm);
}

/* ---- Top-3 -------------------------------------------------------------- */
.tv-rank {
    display: grid;
    grid-template-columns: 34px 1fr auto;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    border: 1px solid var(--tv-border);
    border-radius: 12px;
    background: var(--tv-surface);
    margin-bottom: 10px;
}

.tv-rank.top {
    border-color: var(--tv-accent-line);
    background: var(--tv-accent-soft);
}

.tv-rank-n {
    color: var(--tv-muted);
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    width: 30px;
    height: 30px;
    line-height: 30px;
    border-radius: 8px;
    background: var(--tv-surface-2);
}

.tv-rank.top .tv-rank-n {
    color: #07241D;
    background: var(--tv-accent);
}

.tv-rank-name {
    color: var(--tv-text);
    font-size: 14.5px;
    font-weight: 600;
    margin-bottom: 8px;
}

.tv-rank-cat {
    color: var(--tv-muted);
    font-weight: 400;
    font-size: 12.5px;
    margin-left: 6px;
}

.tv-rank-pct {
    color: var(--tv-text);
    font-size: 15px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* ---- Streamlit components ---------------------------------------------- */
[data-testid="stSpinner"] {
    color: var(--tv-muted);
}

[data-testid="stAlert"] {
    background: var(--tv-surface-2);
    border: 1px solid var(--tv-border);
    border-radius: 12px;
}

[data-testid="stAlert"] * {
    color: var(--tv-text);
}

/* ---- Footer ------------------------------------------------------------- */
.tv-footer {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid var(--tv-border);
    color: var(--tv-muted);
    font-size: 12.5px;
}

/* ---- Responsive --------------------------------------------------------- */
@media (max-width: 820px) {
    .tv-nav {
        display: none;
    }
}

@media (max-width: 640px) {
    .block-container {
        padding: 1.25rem 1rem 2.5rem;
    }

    .tv-hero {
        padding-top: 28px;
    }

    .tv-card {
        padding: 18px;
    }

    .tv-rank {
        grid-template-columns: 30px 1fr;
    }

    .tv-rank-pct {
        grid-column: 2;
    }

    .tv-confidence-value {
        font-size: 32px;
    }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Header and hero
# -----------------------------------------------------------------------------
checkpoint_found = MODEL_PATH.is_file()
status_dot = "tv-dot" if checkpoint_found else "tv-dot warn"
status_text = (
    f"Model checkpoint found · {escape(DEVICE.type.upper())}"
    if checkpoint_found
    else "Model checkpoint missing"
)

render_html(f"""
<div class="tv-header">
    <div class="tv-brand">
        <div class="tv-logo">{BRAND_ICON}</div>
        <div>
            <div class="tv-brand-name">TerraVision AI</div>
            <div class="tv-brand-sub">Geospatial Intelligence</div>
        </div>
    </div>
    <nav class="tv-nav" aria-label="Page sections">
        <a href="#overview">Overview</a>
        <a href="#classification">Classification</a>
        <a href="#about-model">About the Model</a>
    </nav>
    <div class="tv-status" title="Local check of the model file; inference runs on this device.">
        <span class="{status_dot}"></span>{status_text}
    </div>
</div>
""")

render_html(f"""
<div class="tv-hero" id="overview">
    <div class="tv-badge">DEEP LEARNING · EUROSAT RGB</div>
    <div class="tv-hero-title">Understand the Earth, one image at a time.</div>
    <p class="tv-hero-copy">Analyze satellite imagery with deep learning to identify land-cover patterns across ten distinct categories.</p>
    <div class="tv-chips">
        <div class="tv-chip">{icon_svg('<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>', 18)}
            <div><div class="tv-chip-value">10</div><div class="tv-chip-label">Land-cover classes</div></div></div>
        <div class="tv-chip">{icon_svg('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>', 18)}
            <div><div class="tv-chip-value">ResNet18</div><div class="tv-chip-label">Architecture</div></div></div>
        <div class="tv-chip">{icon_svg('<path d="M13 3L5 14h6l-1 7 8-11h-6z"/>', 18)}
            <div><div class="tv-chip-value">PyTorch</div><div class="tv-chip-label">Powered inference</div></div></div>
    </div>
</div>
""")


# -----------------------------------------------------------------------------
# 6. Upload component
# -----------------------------------------------------------------------------
section_header(
    "Classify a satellite image",
    "Upload an image and the model assigns it to one of its ten trained land-cover categories.",
    anchor="classification",
)

format_chips = "".join(
    f'<span class="tv-format">{label}</span>'
    for label in ("JPG", "JPEG", "PNG", "WebP", "JFIF")
)
render_html(f"""
<div class="tv-upload-intro">
    <div class="tv-upload-icon">{icon_svg('<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>', 20)}</div>
    <div>
        <div class="tv-upload-title">Drop an image below, or browse your files</div>
        <div class="tv-upload-copy">Best results come from top-down satellite tiles that show one dominant land-cover type.</div>
        <div class="tv-formats">{format_chips}</div>
    </div>
</div>
""")

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=SUPPORTED_FORMATS,
    label_visibility="collapsed",
)
render_html(
    '<p class="tv-hint">Analysis runs with the local PyTorch model. No external API is used.</p>'
)


# -----------------------------------------------------------------------------
# 7. Empty state and category explorer
# -----------------------------------------------------------------------------
def render_empty_state() -> None:
    render_html(f"""
    <div class="tv-card" style="margin-top:18px;">
        <div class="tv-upload-title">No image analyzed yet</div>
        <div class="tv-upload-copy">Once you upload an image, this area shows the predicted land cover, the model's top probability, and the three most likely categories.</div>
        <div class="tv-steps">
            <div><div class="tv-step-title">1. Upload</div><div class="tv-step-copy">Add a JPG, JPEG, PNG, WebP, or JFIF image.</div></div>
            <div><div class="tv-step-title">2. Analyze</div><div class="tv-step-copy">The image is resized to 224 × 224 and passed to ResNet18.</div></div>
            <div><div class="tv-step-title">3. Review</div><div class="tv-step-copy">See the predicted class and top-3 model probabilities.</div></div>
        </div>
    </div>
    """)


def render_category_explorer() -> None:
    section_header(
        "What can the model recognize?",
        "Ten land-cover and land-use categories from the EuroSAT RGB dataset.",
    )
    cards = "".join(
        f'<div class="tv-cat">'
        f'<div class="tv-cat-icon">{icon_svg(CLASS_ICONS[name], 18)}</div>'
        f'<div class="tv-cat-name">{escape(CLASS_LABELS[name])}</div>'
        f'<div class="tv-cat-copy">{escape(CLASS_DETAILS[name][1])}</div>'
        f"</div>"
        for name in CLASS_NAMES
    )
    render_html(f'<div class="tv-grid">{cards}</div>')


# -----------------------------------------------------------------------------
# 8. Classification results
# -----------------------------------------------------------------------------
def render_results(uploaded_file, image, predictions) -> None:
    best = predictions[0]
    best_label = CLASS_LABELS[best["class"]]
    category, description = CLASS_DETAILS[best["class"]]
    safe_name = escape(uploaded_file.name)

    render_html(
        f'<div class="tv-notice"><span class="tv-dot"></span>'
        f'<span>Image received: <b>{safe_name}</b> ({format_bytes(uploaded_file.size)}). Analysis complete.</span></div>'
    )
    st.write("")

    image_col, result_col = st.columns([1.05, 0.95], gap="large")

    with image_col:
        render_html('<div class="tv-label">Uploaded satellite image</div>')
        with st.container(border=True):
            st.image(image, use_container_width=True)
            render_html(f"""
            <div class="tv-meta">
                <div><span>Filename</span><b>{safe_name}</b></div>
                <div><span>Dimensions</span><b>{image.width} × {image.height} px</b></div>
            </div>
            """)

    with result_col:
        low_confidence_note = ""
        if best["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_note = (
                '<div class="tv-note">The top probability is relatively low. '
                "This image may differ from the EuroSAT-style tiles the model was trained on, "
                "so treat this result with extra caution.</div>"
            )

        render_html(f"""
        <div class="tv-card">
            <div class="tv-label">Predicted land cover</div>
            <div class="tv-result-name">{escape(best_label)}</div>
            <div class="tv-tag">{escape(category)}</div>
            <p class="tv-result-desc">{escape(description)}</p>
            <div class="tv-confidence">
                <div class="tv-confidence-value">{format_percent(best["confidence"])}</div>
                <div class="tv-confidence-caption">Top-class softmax probability</div>
            </div>
            <div class="tv-bar" role="progressbar" aria-label="Top-class probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{best["confidence"]:.2f}">
                <div class="tv-bar-fill" style="width:{bar_width(best["confidence"])}%"></div>
            </div>
            <div class="tv-meta">
                <div><span>Inference device</span><b>{escape(str(DEVICE))}</b></div>
                <div><span>Model input</span><b>224 × 224 RGB</b></div>
            </div>
            {low_confidence_note}
        </div>
        """)

    section_header(
        "Top-3 predictions",
        "Softmax probabilities the model assigned to its three highest-scoring classes. "
        "These are model scores, not verified accuracy.",
    )

    rows = []
    for rank, prediction in enumerate(predictions, start=1):
        name = prediction["class"]
        is_top = rank == 1
        rows.append(
            f'<div class="tv-rank{" top" if is_top else ""}">'
            f'<div class="tv-rank-n">{rank}</div>'
            f"<div>"
            f'<div class="tv-rank-name">{escape(CLASS_LABELS[name])}'
            f'<span class="tv-rank-cat">{escape(CLASS_DETAILS[name][0])}</span></div>'
            f'<div class="tv-bar" role="progressbar" aria-label="Probability for {escape(CLASS_LABELS[name])}" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{prediction["confidence"]:.2f}">'
            f'<div class="tv-bar-fill{"" if is_top else " muted"}" '
            f'style="width:{bar_width(prediction["confidence"])}%"></div></div>'
            f"</div>"
            f'<div class="tv-rank-pct">{format_percent(prediction["confidence"])}</div>'
            f"</div>"
        )
    render_html("".join(rows))

    st.caption(
        "Softmax probability is not a guarantee of correctness. The model was trained on "
        "EuroSAT RGB imagery, so results may be less reliable on substantially different images."
    )


def analyze_upload(uploaded_file):
    """Open the image, load the model, and run inference.
    Returns (image, predictions), or (None, None) after showing a friendly error."""
    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")
    except Exception:
        st.error(
            "This file could not be read as an image. "
            "Try a JPG, JPEG, PNG, WebP, or JFIF file that opens normally on your device."
        )
        return None, None

    try:
        with st.spinner("Analyzing imagery with ResNet18…"):
            model = load_model()
            predictions = predict_image(image, model)
    except FileNotFoundError as error:
        st.error(
            f"{error} Add the checkpoint file and reload the app."
        )
        return None, None
    except Exception as error:
        st.error(
            "The image was read, but the analysis could not be completed. "
            f"Try a different image or restart the app. Details: {error}"
        )
        return None, None

    return image, predictions


analysis_ready = False
if uploaded_file is not None:
    image, predictions = analyze_upload(uploaded_file)
    if image is not None and predictions is not None:
        analysis_ready = True
        render_results(uploaded_file, image, predictions)

if not analysis_ready:
    render_empty_state()
    render_category_explorer()


# -----------------------------------------------------------------------------
# About the model
# -----------------------------------------------------------------------------
section_header(
    "About the model",
    "How TerraVision AI turns an image into a land-cover prediction.",
    anchor="about-model",
)

about_items = [
    (
        "Architecture",
        "ResNet18 convolutional network with its final fully connected layer replaced by a ten-class output layer.",
        '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    ),
    (
        "Training context",
        "Fine-tuned on the EuroSAT RGB dataset of satellite land-use and land-cover images.",
        '<circle cx="12" cy="12" r="8"/><path d="M4 12h16"/><path d="M12 4c-3 3.5-3 12.5 0 16"/><path d="M12 4c3 3.5 3 12.5 0 16"/>',
    ),
    (
        "Image preprocessing",
        "Images are converted to RGB, resized to 224 × 224, and normalized with ImageNet mean and standard deviation.",
        '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/>',
    ),
    (
        "Output",
        "Softmax probabilities across the ten classes. The interface reports the three highest.",
        '<path d="M5 20V10M12 20V4M19 20v-7"/>',
    ),
]
about_cards = "".join(
    f'<div class="tv-cat"><div class="tv-cat-icon">{icon_svg(paths, 18)}</div>'
    f'<div class="tv-cat-name">{title}</div><div class="tv-cat-copy">{copy}</div></div>'
    for title, copy, paths in about_items
)
render_html(f'<div class="tv-grid wide">{about_cards}</div>')
st.caption(
    "Limitations: each image receives a single label, so scenes that mix several land-cover types "
    "may be classified by only one of them. Imagery that differs from EuroSAT in resolution, "
    "scale, or sensor may produce less reliable results."
)


# -----------------------------------------------------------------------------
# 9. Footer
# -----------------------------------------------------------------------------
render_html("""
<div class="tv-footer">
    <span>TerraVision AI · Satellite Image Classification</span>
    <span>Built with PyTorch, torchvision, Pillow &amp; Streamlit</span>
</div>
""")