from __future__ import annotations

import base64
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
    page_title="TDXd–Related Severe Adverse Events (Grade ≥3) Prediction Model",
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
            "Treat as a high-risk patient for grade ≥3 adverse effects; review baseline status, prior treatment lines, metastatic burden, and comorbidities before treatment.",
            "Consider closer monitoring of complete blood count, liver and renal function, electrolytes, infection-related markers, and symptom changes.",
            "If prior severe toxicity, poor performance status, or multi-organ metastasis is present, consider multidisciplinary review and prepare dose adjustment, treatment delay, or supportive-care plans in advance.",
            "Model output is a risk signal only; final management should be based on clinical judgment, drug labeling, and guideline recommendations.",
        ]
    if prob >= LOW_RISK_CUTOFF:
        return [
            "The model suggests a moderate risk of grade ≥3 adverse effects; consider moderately intensified follow-up beyond routine monitoring.",
            "Review key risk factors before treatment, including recent laboratory results and prior treatment tolerance.",
            "Educate the patient about warning signs; advise prompt medical contact for fever, marked fatigue, respiratory symptoms, or gastrointestinal symptoms.",
            "The model supports risk stratification and should not be used alone to modify treatment plans.",
        ]
    return [
        "The model suggests a lower risk of grade ≥3 adverse effects; routine monitoring may be appropriate.",
        "Standard patient education and periodic laboratory monitoring are still recommended.",
        "Reassess risk if the patient’s clinical status changes.",
        "Low risk does not mean no risk; interpret the model output together with clinical judgment.",
    ]


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
    st.subheader("Enter the 8 patient variables")
    train_x, _ = load_training_data()
    defaults = train_x.median(numeric_only=True)

    with st.form("single_patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ki67 = st.number_input("Ki-67", min_value=0.0, max_value=100.0, value=float(defaults["Ki-67"]), step=1.0)
            pr = st.number_input("PR (%)", min_value=0.0, max_value=100.0, value=float(defaults["PR (%)"]), step=1.0)
            her2_lines = st.number_input("HER2 lines", min_value=0.0, max_value=10.0, value=float(defaults["HER2 lines"]), step=1.0)
        with col2:
            er = st.selectbox(
                "ER",
                options=[0, 1],
                index=int(round(defaults["ER"])),
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            lung = st.selectbox("Lung", options=[0, 1], index=int(round(defaults["Lung"])))
            her2_adc = st.selectbox("HER2 ADC", options=[0, 1], index=int(round(defaults["HER2 ADC"])))
        with col3:
            viscral = st.selectbox("Viscral", options=[0, 1], index=int(round(defaults["Viscral"])))
            nvb = st.selectbox("NVB", options=[0, 1], index=int(round(defaults["NVB"])))
            submitted = st.form_submit_button("Predict Grade ≥3 Adverse Effect Risk", use_container_width=True)

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
    st.metric("Final ensemble probability", f"{prob:.1%}")

    if risk_color(prob) == "red":
        st.error(f"Prediction result：{level}")
    elif risk_color(prob) == "orange":
        st.warning(f"Prediction result：{level}")
    else:
        st.success(f"Prediction result：{level}")

    st.markdown("### Clinical guidance")
    for item in clinical_guidance(prob):
        st.markdown(f"- {item}")

    st.markdown("### Local contribution hints for the current patient (quick approximation)")
    st.caption("Note: This is not a real-time SHAP calculation. It replaces one variable at a time with the training-cohort reference value and observes the probability change for quick online guidance.")
    contrib = local_perturbation_contribution(patient)
    st.dataframe(contrib, use_container_width=True, hide_index=True)
    st.bar_chart(contrib.set_index("Feature")["Contribution_Approx"])


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
    st.title("TDXd–Related Severe Adverse Events (Grade ≥3) Prediction Model")

    st.sidebar.header("Model notes")
    st.sidebar.info(
        f"After entering Ki-67, PR (%), HER2 lines, ER, Lung, HER2 ADC, Viscral, and NVB, "
        f"the system outputs the predicted probability of AE3_All positivity. "
        f"The current high-risk cutoff is the internal Youden threshold: {RISK_THRESHOLD:.1%}."
    )
    st.sidebar.markdown(
        """
        **Risk group**

        - Low risk: < 15%
        - Intermediate risk: 15% to the internal Youden threshold (20.5%)
        - High risk: ≥ the internal Youden threshold (20.5%)
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
