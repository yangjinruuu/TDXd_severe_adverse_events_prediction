from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"

FEATURES = ["Ki-67", "PR (%)", "HER2 lines", "ER", "Lung", "HER2 ADC", "Viscral", "NVB"]
CONTINUOUS = ["Ki-67", "PR (%)", "HER2 lines"]
BINARY = ["ER", "Lung", "HER2 ADC", "Viscral", "NVB"]
OUTCOME = "AE3_All"

# From project performance file: internal Youden threshold.
RISK_THRESHOLD = 0.205443
LOW_RISK_CUTOFF = 0.15
HIGH_RISK_CUTOFF = RISK_THRESHOLD


st.set_page_config(
    page_title="TDXd–Related Severe Adverse Events Prediction Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        .risk-card {
            padding: 1rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid #e6e6e6;
            background: #ffffff;
            box-shadow: 0 1px 8px rgba(0,0,0,0.04);
        }
        .small-note {color: #666; font-size: 0.92rem;}
        .page-subtitle {color: #7a7f87; font-size: 0.95rem; margin-top: -0.65rem; margin-bottom: 1.4rem;}
        .probability-label {color: #3d4149; font-size: 1.02rem; margin-top: 0.35rem; margin-bottom: 0.25rem;}
        .probability-value {font-size: 2.6rem; font-weight: 650; color: #2f3440; line-height: 1.05; margin-top: 0.25rem; margin-bottom: 1rem;}
        .risk-banner {padding: 0.85rem 1rem; border-radius: 0.55rem; font-size: 1.12rem; font-weight: 650; margin: 0.7rem 0 1.3rem 0;}
        .risk-low {background: #eaf6ec; color: #2e7d32;}
        .risk-intermediate {background: #fffbe0; color: #8a6d00;}
        .risk-high {background: #fdeaea; color: #b3261e;}
        .guidance-heading {font-size: 1.35rem; font-weight: 700; margin-top: 1rem; margin-bottom: 0.8rem;}
        .guidance-risk {font-size: 1.02rem; font-weight: 650; margin-bottom: 0.45rem;}
        .guidance-content {font-size: 0.96rem; line-height: 1.3;}
        .guidance-content h4 {font-size: 1rem; margin: 0.55rem 0 0.12rem 0;}
        .guidance-content ul {margin: 0.05rem 0 0.35rem 1.35rem; padding-left: 0.8rem;}
        .guidance-content li {margin: 0.08rem 0; padding-left: 0.1rem;}
        .about-card {padding: 0.95rem 1rem; border-radius: 0.7rem; background: #e8f1fc; color: #2f618e; line-height: 1.55; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    x = df.rename(columns={"Viscral ": "Viscral"}).copy()
    missing = [c for c in FEATURES if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required variables: {', '.join(missing)}")
    x = x[FEATURES].copy()
    for col in FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    ref = reference if reference is not None else x
    for col in CONTINUOUS:
        fill_value = pd.to_numeric(ref[col], errors="coerce").median()
        x[col] = x[col].fillna(fill_value)

    for col in BINARY:
        ref_col = pd.to_numeric(ref[col], errors="coerce")
        mode = ref_col.mode(dropna=True)
        fill_value = float(mode.iloc[0]) if len(mode) else 0.0
        x[col] = x[col].fillna(fill_value)
        x[col] = (x[col] >= 0.5).astype(float)

    return x.astype(float)


@st.cache_data(show_spinner=False)
def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = DATA_DIR / "train_data_ae3_all.csv"
    df = pd.read_csv(data_path)
    x = prepare_features(df)
    y = (pd.to_numeric(df[OUTCOME], errors="coerce") >= 0.5).astype(int)
    return x, y


@st.cache_resource(show_spinner=False)
def build_model():
    x, y = load_training_data()
    scale_center = x.mean(axis=0)
    scale_scale = x.std(axis=0, ddof=1).replace(0, 1.0)
    x_scaled = (x - scale_center) / scale_scale

    tree = DecisionTreeClassifier(max_depth=2, random_state=123)
    try:
        ada = AdaBoostClassifier(estimator=tree, n_estimators=20, random_state=123)
    except TypeError:
        ada = AdaBoostClassifier(base_estimator=tree, n_estimators=20, random_state=123)
    svc = SVC(C=35, gamma=0.005, kernel="rbf", probability=True, random_state=123)
    svm_rbf = SVC(C=5, gamma=0.005, kernel="rbf", probability=True, random_state=123)

    ada.fit(x, y)
    svc.fit(x_scaled, y)
    svm_rbf.fit(x_scaled, y)
    return ada, svc, svm_rbf, scale_center, scale_scale


def predict_probabilities(input_df: pd.DataFrame) -> pd.DataFrame:
    train_x, _ = load_training_data()
    x = prepare_features(input_df, reference=train_x)
    ada, svc, svm_rbf, scale_center, scale_scale = build_model()
    x_scaled = (x - scale_center) / scale_scale
    out = pd.DataFrame(index=x.index)
    out["AdaBoost"] = ada.predict_proba(x)[:, 1]
    out["SVC"] = svc.predict_proba(x_scaled)[:, 1]
    out["SVM_RBF"] = svm_rbf.predict_proba(x_scaled)[:, 1]
    out["Final_Ensemble"] = out[["AdaBoost", "SVC", "SVM_RBF"]].mean(axis=1)
    out["Risk_Level"] = out["Final_Ensemble"].apply(risk_level)
    out["Risk_Threshold"] = RISK_THRESHOLD
    return out


def risk_level(prob: float) -> str:
    if prob >= HIGH_RISK_CUTOFF:
        return "High risk"
    if prob >= LOW_RISK_CUTOFF:
        return "Intermediate risk"
    return "Low risk"


def risk_color(prob: float) -> str:
    if prob >= HIGH_RISK_CUTOFF:
        return "red"
    if prob >= LOW_RISK_CUTOFF:
        return "orange"
    return "green"


def clinical_guidance(prob: float) -> list[str]:
    if prob >= HIGH_RISK_CUTOFF:
        return [
            "High-Risk Patients – Immediate Clinical Attention",
            "Monitoring & Prevention: Check complete blood count and relevant laboratory tests before each treatment cycle and consider twice-weekly monitoring when clinically indicated.",
            "Monitoring & Prevention: Monitor temperature, infection symptoms, bleeding, severe fatigue, respiratory symptoms, and other clinically significant changes.",
            "Monitoring & Prevention: Review baseline organ function, concomitant medications, prior treatment toxicity, and other patient-specific risk factors.",
            "Treatment Adjustments: Consider dose delay, dose modification, supportive treatment, or multidisciplinary review according to the product label and institutional guidance.",
            "Treatment Adjustments: Evaluate the need for prompt clinical assessment or hospital referral if severe or rapidly worsening symptoms occur.",
            "Patient Education: Explain warning symptoms and provide clear instructions for contacting the treatment team or seeking urgent care.",
            "Patient Education: Reinforce infection-prevention measures, medication adherence, and the importance of reporting new symptoms promptly.",
        ]
    if prob >= LOW_RISK_CUTOFF:
        return [
            "Intermediate-Risk Patients – Enhanced Surveillance",
            "Monitoring: Check complete blood count and relevant laboratory tests before each treatment cycle and consider additional follow-up when clinically appropriate.",
            "Monitoring: Monitor temperature, infection symptoms, unusual bleeding, severe fatigue, respiratory symptoms, and persistent gastrointestinal symptoms.",
            "Prevention: Review recent laboratory results, prior treatment tolerance, infection risk, nutritional status, and other patient-specific risk factors.",
            "Prevention: Reinforce infection-prevention measures and address modifiable risk factors when appropriate.",
            "Follow-up: Arrange follow-up before the next treatment cycle and reassess the risk if the patient’s clinical condition changes.",
            "Follow-up: Review treatment tolerance and laboratory trends at each visit.",
            "Patient Education: Explain warning signs and when to contact the treatment team for further assessment.",
        ]
    return [
        "Low-Risk Patients – Routine Monitoring",
        "Standard Care: Continue routine complete blood count and laboratory monitoring according to the treatment protocol and product labeling.",
        "Standard Care: Check relevant laboratory results before each treatment cycle and maintain standard precautions.",
        "Patient Education: Provide standard education about possible adverse events, infection prevention, and when to contact the treatment team.",
        "Patient Education: Encourage patients to report fever, infection symptoms, unusual bleeding, severe fatigue, respiratory symptoms, or other new concerns.",
        "Follow-up: Maintain scheduled clinical follow-up and review treatment tolerance at each visit.",
        "Follow-up: Reassess risk if new symptoms, laboratory abnormalities, treatment changes, or clinical deterioration occur.",
    ]


def risk_presentation(prob: float) -> tuple[str, str, str, str]:
    if prob >= HIGH_RISK_CUTOFF:
        return "⚠️", "High Risk", "risk-high", "🔴"
    if prob >= LOW_RISK_CUTOFF:
        return "⚠️", "Intermediate Risk", "risk-intermediate", "🟡"
    return "✅", "Low Risk", "risk-low", "🟢"


def yes_no_label(value: int | float) -> str:
    return "Yes" if int(value) == 1 else "No"


def local_perturbation_contribution(input_df: pd.DataFrame) -> pd.DataFrame:
    train_x, _ = load_training_data()
    base_prob = float(predict_probabilities(input_df)["Final_Ensemble"].iloc[0])
    reference_values = {}
    for col in FEATURES:
        if col in BINARY:
            reference_values[col] = float(train_x[col].mode(dropna=True).iloc[0])
        else:
            reference_values[col] = float(train_x[col].median())

    rows = []
    for feature in FEATURES:
        altered = input_df.copy()
        altered.loc[altered.index[0], feature] = reference_values[feature]
        altered_prob = float(predict_probabilities(altered)["Final_Ensemble"].iloc[0])
        rows.append(
            {
                "Feature": feature,
                "Input_Value": float(input_df.iloc[0][feature]),
                "Reference_Value": reference_values[feature],
                "Contribution_Approx": base_prob - altered_prob,
            }
        )
    return pd.DataFrame(rows).sort_values("Contribution_Approx", key=lambda s: s.abs(), ascending=False)


def pdf_download_or_embed(path: Path, height: int = 720) -> None:
    if not path.exists():
        st.info(f"Figure file not found：{path.name}")
        return
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    st.download_button(
        label=f"Download {path.name}",
        data=data,
        file_name=path.name,
        mime="application/pdf",
        use_container_width=True,
    )
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )


def input_form() -> pd.DataFrame | None:
    st.subheader("Please enter the patient parameters")
    train_x, _ = load_training_data()
    defaults = train_x.median(numeric_only=True)

    with st.form("single_patient_form"):
        basic_col, treatment_col, metastasis_col = st.columns(3)
        with basic_col:
            st.markdown("#### Basic characteristics")
            er = st.selectbox("ER", options=[0, 1], index=int(round(defaults["ER"])), format_func=yes_no_label)
            pr = st.number_input("PR (%)", min_value=0.0, max_value=100.0, value=float(defaults["PR (%)"]), step=1.0)
            ki67 = st.number_input("Ki-67", min_value=0.0, max_value=100.0, value=float(defaults["Ki-67"]), step=1.0)
        with treatment_col:
            st.markdown("#### Treatment-related factors")
            nvb = st.selectbox("NVB", options=[0, 1], index=int(round(defaults["NVB"])), format_func=yes_no_label)
            her2_lines = st.number_input("HER2 lines", min_value=0.0, max_value=10.0, value=float(defaults["HER2 lines"]), step=1.0)
            her2_adc = st.selectbox("HER2 ADC", options=[0, 1], index=int(round(defaults["HER2 ADC"])), format_func=yes_no_label)
        with metastasis_col:
            st.markdown("#### Metastasis sites")
            lung = st.selectbox("Lung", options=[0, 1], index=int(round(defaults["Lung"])), format_func=yes_no_label)
            viscral = st.selectbox("Viscral", options=[0, 1], index=int(round(defaults["Viscral"])), format_func=yes_no_label)
            st.markdown("")
            submitted = st.form_submit_button("Predict Severe Adverse Events Risk", use_container_width=True)

    if not submitted:
        return None
    return pd.DataFrame(
        [
            {
                "Ki-67": ki67,
                "PR (%)": pr,
                "HER2 lines": her2_lines,
                "ER": er,
                "Lung": lung,
                "HER2 ADC": her2_adc,
                "Viscral": viscral,
                "NVB": nvb,
            }
        ]
    )


def page_single_patient() -> None:
    patient = input_form()
    if patient is None:
        st.info("Enter the variables and click Predict to display risk probabilities, risk stratification, and guidance.")
        return

    pred = predict_probabilities(patient)
    prob = float(pred["Final_Ensemble"].iloc[0])
    level = risk_level(prob)

    st.markdown("---")
    icon, risk_text, risk_class, risk_ball = risk_presentation(prob)
    st.markdown("### Prediction Result")
    st.markdown("<div class='probability-label'>Probability of TDXd-related severe adverse events</div>", unsafe_allow_html=True)
    st.progress(min(max(prob, 0.0), 1.0))
    st.markdown(f"<div class='probability-value'>{prob:.1%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='risk-banner {risk_class}'>{icon} {risk_text}</div>", unsafe_allow_html=True)

    guidance = clinical_guidance(prob)
    st.markdown("<div class='guidance-heading'>Clinical Practice Recommendations</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='guidance-risk'>{risk_ball} {guidance[0]}</div>", unsafe_allow_html=True)
    guidance_html = ["<div class='guidance-content'>"]
    previous_heading = None
    for item in guidance[1:]:
        if ":" in item:
            heading, detail = item.split(":", 1)
            if heading == previous_heading:
                guidance_html.append(f"<li>{html.escape(detail.strip())}</li>")
            else:
                if previous_heading is not None:
                    guidance_html.append("</ul>")
                guidance_html.append(f"<h4>{html.escape(heading)}:</h4><ul><li>{html.escape(detail.strip())}</li>")
            previous_heading = heading
        else:
            guidance_html.append(f"<p>• {html.escape(item)}</p>")
            previous_heading = None
    if previous_heading is not None:
        guidance_html.append("</ul>")
    guidance_html.append("</div>")
    st.markdown("".join(guidance_html), unsafe_allow_html=True)

def page_global_explanation() -> None:
    st.subheader("Global explanations and model evidence")
    st.caption("The following high-resolution PDFs were generated in the project and are bundled with the app.")

    options = {
        "SHAP global importance combined plot": ASSET_DIR / "global/SHAP_Global_Importance_Bar_Rose_Beeswarm_Combined.pdf",
        "SHAP values and percentages combined plot": ASSET_DIR / "global/SHAP_combined_with_values_and_percentages.pdf",
        "SHAP beeswarm plot": ASSET_DIR / "global/SHAP_Beeswarm_Final_Ensemble.pdf",
        "SZL vs ZN explanation comparison": ASSET_DIR / "global/SZL_ZN_Explanation_Comparison_Combined.pdf",
        "1D PDP rug plot": ASSET_DIR / "pdp/PDP_Reference_Rug_8_Combined_2x4.pdf",
        "High- and low-risk SHAP waterfall plots": ASSET_DIR / "local/High_Low_Risk_SHAP_Waterfall_Combined.pdf",
        "LIME-style local explanations": ASSET_DIR / "local/LIME_Local_Explanations_3_Cases.pdf",
    }
    selected = st.selectbox("Select an explanation figure", list(options))
    pdf_download_or_embed(options[selected])


def page_batch_prediction() -> None:
    st.subheader("Batch prediction and download")
    template_path = DATA_DIR / "sample_batch_template.csv"
    st.download_button(
        "Download batch prediction template CSV",
        data=template_path.read_bytes(),
        file_name="sample_batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    uploaded = st.file_uploader("Upload a CSV file containing the 8 variable columns", type=["csv"])
    if uploaded is None:
        st.info("After upload, the app will output three base-model probabilities, the final ensemble probability, and risk group.")
        return

    try:
        df = pd.read_csv(uploaded)
        pred = predict_probabilities(df)
        result = pd.concat([prepare_features(df, reference=load_training_data()[0]).reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction results CSV",
            data=result.to_csv(index=False, encoding="utf-8-sig"),
            file_name="AE3_All_batch_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Batch prediction failed：{exc}")


def page_deployment_notes() -> None:
    st.subheader("Deployment and usage notes")
    st.markdown(
        """
        **Purpose**: This app predicts the risk of grade ≥3 adverse effects (AE3_All) using 8 patient variables.

        **Deployment workflow**:

        1. Open this app folder locally.
        2. Install dependencies: `pip install -r requirements.txt`
        3. Run locally: `streamlit run app.py`
        4. After checking the app, upload this folder to GitHub.
        5. In Streamlit Community Cloud, select the repository and set the main file path to `app.py`.

        **Important note**: This model is a clinical decision-support tool and does not replace physician judgment. Any treatment adjustment should consider the patient’s clinical context, drug labeling, guidelines, and multidisciplinary discussion.
        """
    )


def main() -> None:
    inject_style()
    st.title("TDXd–Related Severe Adverse Events Prediction Model")
    st.markdown("<div class='page-subtitle'>Predicting severe adverse events (Grade ≥3) in patients receiving trastuzumab deruxtecan (T-DXd) therapy</div>", unsafe_allow_html=True)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "<div class='about-card'>After entering the relevant patient parameters, the system estimates the probability of TDXd-related severe adverse events. The internal Youden index threshold for high-risk classification is 20.5%.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Risk level description")
    st.sidebar.markdown(
        """
        - **Low risk:** < 15%
        - **Intermediate risk:** 15% - 20.5%
        - **High risk:** ≥ 20.5%
        """
    )

    tabs = st.tabs(["Single-patient prediction", "Global explanation page", "Batch prediction and download", "Deployment notes"])
    with tabs[0]:
        page_single_patient()
    with tabs[1]:
        page_global_explanation()
    with tabs[2]:
        page_batch_prediction()
    with tabs[3]:
        page_deployment_notes()


if __name__ == "__main__":
    main()
