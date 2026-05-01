#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

st.set_page_config(
    page_title="PhytoLike | Phytochemical-Likeness Prediction Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DEFAULT = "phytochemical_likeness_phase1_v2_model.joblib"
APP_TITLE = "PhytoLike"
APP_SUBTITLE = "Phytochemical-Likeness Prediction Platform"
APP_TAGLINE = (
    "AI-enabled prediction of whether a small molecule lies closer to plant-derived "
    "phytochemical space."
)
BASE_DIR = Path(__file__).resolve().parent
HOME_ILLUSTRATION = BASE_DIR / "phytolike_home_illustration.svg"

st.markdown(
    """
    <style>
    :root {
        --green-dark: #0f5b38;
        --green-main: #198754;
        --green-light: #eef8f3;
        --green-border: #d6eadf;
        --text-dark: #18382a;
        --text-soft: #61756a;
        --shadow: 0 10px 28px rgba(17, 38, 28, 0.08);
    }

    .stApp {
        background: #f2faf5;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.15rem;
        padding-left: 1.8rem;
        padding-right: 1.8rem;
        padding-bottom: 1.6rem;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.2rem;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 1rem;
        font-weight: 650;
    }

    .hero {
        background: linear-gradient(135deg, #0f5c38 0%, #17824f 55%, #2d9b69 100%);
        border-radius: 28px;
        padding: 2.2rem 2.6rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .hero:before {
        content: "";
        position: absolute;
        right: -50px;
        top: -40px;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
    }

    .hero:after {
        content: "";
        position: absolute;
        right: 180px;
        bottom: -55px;
        width: 170px;
        height: 170px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }

    .hero-pill {
        display: inline-block;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: 3.15rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.45rem;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        font-size: 1.28rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        opacity: 0.98;
    }

    .hero-text {
        font-size: 1.12rem;
        line-height: 1.7;
        max-width: 1080px;
        color: rgba(255,255,255,0.96);
        margin: 0 auto;
    }

    .hero-inner {
        position: relative;
        z-index: 2;
        text-align: center;
        max-width: 1100px;
        margin: 0 auto;
    }


    .card {
        background: linear-gradient(180deg, #ffffff 0%, #fbfefc 100%);
        border: 1px solid var(--green-border);
        border-radius: 22px;
        padding: 1.7rem 1.6rem 1.45rem 1.6rem;
        box-shadow: var(--shadow);
        height: 100%;
        min-height: 270px;
    }

    .card-title {
        font-size: 1.32rem;
        font-weight: 750;
        color: var(--text-dark);
        margin-bottom: 0.55rem;
    }

    .card-text {
        color: var(--text-soft);
        font-size: 1.18rem;
        line-height: 1.9;
    }

    .metric-shell {
        background: linear-gradient(180deg, #ffffff 0%, #fbfefc 100%);
        border: 1px solid var(--green-border);
        border-radius: 18px;
        padding: 0.95rem 1rem 0.85rem 1rem;
        box-shadow: var(--shadow);
        height: 100%;
    }

    .metric-label {
        color: var(--text-soft);
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: var(--green-dark);
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.15;
        word-break: break-word;
    }

    .metric-note {
        color: var(--text-soft);
        font-size: 0.84rem;
        margin-top: 0.3rem;
    }

    .footer-note {
        text-align: center;
        font-size: 0.9rem;
        color: #72827a;
        margin-top: 1rem;
        padding-bottom: 0.5rem;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(model_path: str):
    return load(model_path)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles) if smiles else None


def featurize_mol(mol: Chem.Mol, radius: int, nbits: int):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    desc = np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
        sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
    ], dtype=np.float32)
    return fp, np.concatenate([arr, desc], axis=0)


def label_from_score(score: float) -> str:
    if score >= 0.80:
        return "Highly phytochemical-like"
    if score >= 0.60:
        return "Moderately phytochemical-like"
    if score >= 0.40:
        return "Borderline phytochemical-like"
    return "Weakly phytochemical-like"


def confidence_from_similarity(sim: float) -> str:
    if sim >= 0.70:
        return "High"
    if sim >= 0.50:
        return "Moderate"
    return "Low"


def nearest_neighbors(query_fp, train_fps, train_meta, top_k=5):
    sims = []
    for i, fp in enumerate(train_fps):
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        sims.append((i, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    rows = []
    for idx, sim in sims[:top_k]:
        row = dict(train_meta[idx])
        row["similarity"] = round(float(sim), 4)
        rows.append(row)
    best = float(sims[0][1]) if sims else 0.0
    return best, pd.DataFrame(rows)


def draw_molecule(smiles: str, size=(420, 260)):
    # Disabled in deployment because RDKit drawing may require system GUI
    # libraries (e.g. libXrender) that are unavailable on some Streamlit
    # cloud environments. The prediction workflow does not depend on drawing.
    return None


def predict_df(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model = artifact["model"]
    radius = int(artifact.get("fingerprint_radius", 2))
    nbits = int(artifact.get("fingerprint_nbits", 2048))
    train_fps = artifact.get("training_fingerprints", [])
    train_meta = artifact.get("training_metadata", [])
    rows = []
    for _, row in df.iterrows():
        compound_id = row.get("compound_id", "")
        compound_name = row.get("compound_name", "")
        input_smiles = row.get("smiles", "")
        canonical = canonicalize_smiles(input_smiles)
        out = {
            "compound_id": compound_id,
            "compound_name": compound_name,
            "input_smiles": input_smiles,
            "canonical_smiles": canonical,
            "is_valid_smiles": canonical is not None,
            "phytochemical_likeness_score": np.nan,
            "prediction": "invalid_smiles",
            "nearest_neighbor_similarity": np.nan,
            "confidence": "NA",
        }
        if canonical is not None:
            mol = mol_from_smiles(canonical)
            fp_obj, feat = featurize_mol(mol, radius, nbits)
            score = float(model.predict_proba(feat.reshape(1, -1))[:, 1][0])
            out["phytochemical_likeness_score"] = score
            out["prediction"] = label_from_score(score)
            if train_fps:
                best_sim, _ = nearest_neighbors(fp_obj, train_fps, train_meta, top_k=5)
                out["nearest_neighbor_similarity"] = best_sim
                out["confidence"] = confidence_from_similarity(best_sim)
        rows.append(out)
    return pd.DataFrame(rows)


def card(title: str, text: str):
    st.markdown(
        f"<div class='card'><div class='card-title'>{title}</div><div class='card-text'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def metric_box(label: str, value: str, note: str = ""):
    st.markdown(
        f"<div class='metric-shell'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


model_path = MODEL_DEFAULT
if not Path(model_path).exists():
    st.error(f"Model file not found: {model_path}. Please train the model first using `train_phytochemical_likeness_phase1_v2.py`.")
    st.stop()
artifact = load_artifact(model_path)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-inner">
            <div class="hero-pill">Web platform for phytochemical-likeness prediction</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            <div class="hero-text">{APP_TAGLINE}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

home_tab, prediction_tab, tutorial_tab, contact_tab = st.tabs([
    "🏠 Home", "🧪 Prediction", "📘 Tutorial", "📩 Contact"
])

with home_tab:
    top_left, top_right = st.columns([1.12, 0.98], gap="large")
    with top_left:
        card(
            "About the platform",
            "PhytoLike predicts phytochemical-likeness from molecular structure and helps prioritize compounds that are chemically closer to plant-derived phytochemical space. It can serve as an upstream filter before downstream discovery analyses."
        )
        st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)
        bottom_left, bottom_right = st.columns(2, gap="large")
        
        st.markdown("<div style='height: 2.6rem;'></div>", unsafe_allow_html=True)
    with top_right:
        if Path(HOME_ILLUSTRATION).exists():
            st.image(HOME_ILLUSTRATION, use_container_width=True)
        else:
            card("Platform illustration", "A visual overview of the phytochemical-likeness workflow.")

with prediction_tab:
    st.markdown("### Prediction workspace")
    st.caption("Predict phytochemical-likeness for a single compound or perform batch analysis.")
    single_pred_tab, batch_pred_tab = st.tabs(["Single compound", "Batch CSV"])

    with single_pred_tab:
        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            smiles = st.text_area(
                "Enter SMILES",
                value="O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
                height=110,
            )
            compound_name = st.text_input("Compound name", value="Quercetin")
            run_single = st.button("Predict compound", type="primary", use_container_width=True)
        with right:
            canonical_preview = canonicalize_smiles(smiles)
            if canonical_preview:
                st.success("Valid SMILES detected")
                st.markdown("**Canonical SMILES**")
                st.code(canonical_preview, language=None)
            else:
                st.info("Enter a valid SMILES string to view the canonical form.")

        if run_single:
            query_df = pd.DataFrame([
                {"compound_id": "Q1", "compound_name": compound_name, "smiles": smiles}
            ])
            pred = predict_df(query_df, artifact)
            row = pred.iloc[0]
            if not row["is_valid_smiles"]:
                st.error("Invalid SMILES. Please check the input and try again.")
            else:
                m1, m2, m3 = st.columns(3, gap="large")
                with m1:
                    metric_box("Phytochemical-likeness score", f"{row['phytochemical_likeness_score']:.3f}", "Continuous prediction score")
                with m2:
                    metric_box("Prediction", row["prediction"], "Categorical interpretation")
                with m3:
                    metric_box("Confidence", row["confidence"], f"Nearest-neighbor similarity: {row['nearest_neighbor_similarity']:.3f}")

                st.markdown("#### Prediction details")
                show_df = pred.copy()
                show_df["phytochemical_likeness_score"] = show_df["phytochemical_likeness_score"].round(3)
                show_df["nearest_neighbor_similarity"] = show_df["nearest_neighbor_similarity"].round(3)
                st.dataframe(show_df, use_container_width=True, hide_index=True)

                if artifact.get("training_fingerprints"):
                    mol = mol_from_smiles(row["canonical_smiles"])
                    fp_obj, _ = featurize_mol(
                        mol,
                        int(artifact.get("fingerprint_radius", 2)),
                        int(artifact.get("fingerprint_nbits", 2048)),
                    )
                    _, nn_df = nearest_neighbors(
                        fp_obj,
                        artifact.get("training_fingerprints", []),
                        artifact.get("training_metadata", []),
                        top_k=5,
                    )
                    st.markdown("#### Nearest training neighbors")
                    st.dataframe(nn_df, use_container_width=True, hide_index=True)

    with batch_pred_tab:
        st.markdown("#### Batch prediction")
        st.caption("Upload a CSV with columns such as `compound_id`, `compound_name`, and `smiles`.")
        demo_csv = (
            "compound_id,compound_name,smiles\n"
            "Q1,Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12\n"
            "Q2,Caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O\n"
            "Q3,Curcumin,COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O\n"
        )
        d1, d2 = st.columns([0.28, 0.72], gap="large")
        with d1:
            st.download_button(
                "Download demo CSV",
                data=demo_csv,
                file_name="phytolike_demo_queries.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            uploaded = st.file_uploader("Upload query CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            if "smiles" not in batch_df.columns:
                st.error("The uploaded CSV must contain a `smiles` column.")
            else:
                pred = predict_df(batch_df, artifact)
                show_pred = pred.copy()
                show_pred["phytochemical_likeness_score"] = show_pred["phytochemical_likeness_score"].round(3)
                show_pred["nearest_neighbor_similarity"] = show_pred["nearest_neighbor_similarity"].round(3)
                st.success(f"Prediction completed for {len(show_pred)} compounds.")
                st.dataframe(show_pred, use_container_width=True, hide_index=True)
                csv_bytes = pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions",
                    data=csv_bytes,
                    file_name="phytolike_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with tutorial_tab:
    st.markdown("### Quick tutorial")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        card(
            "How to use",
            "Go to the Prediction tab, enter a SMILES string or upload a CSV file, and run the analysis. The platform returns a numerical score, a prediction category, and a confidence estimate."
        )
    with c2:
        card(
            "How to read the output",
            "Higher scores indicate stronger resemblance to plant-derived phytochemical space. Confidence reflects similarity to known training molecules and helps judge prediction reliability."
        )

    st.markdown("### Score interpretation")
    s1, s2, s3, s4 = st.columns(4, gap="large")
    with s1:
        metric_box("0.80 – 1.00", "Highly", "Strong resemblance to phytochemical space")
    with s2:
        metric_box("0.60 – 0.79", "Moderate", "Reasonably phytochemical-like")
    with s3:
        metric_box("0.40 – 0.59", "Borderline", "Intermediate or uncertain")
    with s4:
        metric_box("< 0.40", "Weak", "Less likely to be phytochemical-like")

with contact_tab:
    st.markdown("### Contact")
    card(
        "Principal Investigator",
        "Sneha Murmu, Ph.D.<br>Scientist (Bioinformatics)<br>Division of Agricultural Bioinformatics<br>ICAR-Indian Agricultural Statistics Research Institute, New Delhi<br>Email: murmu.sneha07@gmail.com"
    )

st.markdown(
    "<div class='footer-note'>PhytoLike • AI-enabled platform for phytochemical-likeness prediction</div>",
    unsafe_allow_html=True,
)
