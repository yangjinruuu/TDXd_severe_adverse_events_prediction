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
    page_title="严重不良反应预测模型",
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
        raise ValueError(f"缺少必要变量: {', '.join(missing)}")
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
        return "高风险"
    if prob >= LOW_RISK_CUTOFF:
        return "中风险"
    return "低风险"


def risk_color(prob: float) -> str:
    if prob >= HIGH_RISK_CUTOFF:
        return "red"
    if prob >= LOW_RISK_CUTOFF:
        return "orange"
    return "green"


def clinical_guidance(prob: float) -> list[str]:
    if prob >= HIGH_RISK_CUTOFF:
        return [
            "建议作为严重不良反应高风险患者进行重点管理，治疗前完成基础状态、既往治疗线数、转移负荷和合并症复核。",
            "建议加强血常规、肝肾功能、电解质、感染相关指标和症状评估的监测频率。",
            "若患者存在既往高危毒性、体能状态较差或多器官转移，建议多学科讨论并提前制定剂量调整、延迟给药或支持治疗预案。",
            "模型结果仅作为风险提示，最终处理应结合临床判断、药品说明书和指南要求。",
        ]
    if prob >= LOW_RISK_CUTOFF:
        return [
            "提示存在一定严重不良反应风险，建议按常规频率基础上适度加强随访。",
            "治疗前复核关键危险因素，关注近期实验室指标和既往治疗耐受性。",
            "建议向患者说明可能不良反应信号，出现发热、明显乏力、呼吸道症状或消化道症状时及时就诊。",
            "模型结果用于辅助分层，不应单独作为调整治疗方案的依据。",
        ]
    return [
        "当前模型提示严重不良反应风险较低，可按常规流程监测。",
        "仍建议进行标准化用药宣教和周期性实验室检查。",
        "若后续出现新的临床状态变化，应重新评估风险。",
        "低风险不代表无风险，模型结果需结合临床判断使用。",
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
        st.info(f"暂未找到图形文件：{path.name}")
        return
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    st.download_button(
        label=f"下载 {path.name}",
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
    st.subheader("请输入患者 8 个变量")
    train_x, _ = load_training_data()
    defaults = train_x.median(numeric_only=True)

    with st.form("single_patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ki67 = st.number_input("Ki-67", min_value=0.0, max_value=100.0, value=float(defaults["Ki-67"]), step=1.0)
            pr = st.number_input("PR (%)", min_value=0.0, max_value=100.0, value=float(defaults["PR (%)"]), step=1.0)
            her2_lines = st.number_input("HER2 lines", min_value=0.0, max_value=10.0, value=float(defaults["HER2 lines"]), step=1.0)
        with col2:
            er = st.selectbox("ER", options=[0, 1], index=int(round(defaults["ER"])))
            lung = st.selectbox("Lung", options=[0, 1], index=int(round(defaults["Lung"])))
            her2_adc = st.selectbox("HER2 ADC", options=[0, 1], index=int(round(defaults["HER2 ADC"])))
        with col3:
            viscral = st.selectbox("Viscral", options=[0, 1], index=int(round(defaults["Viscral"])))
            nvb = st.selectbox("NVB", options=[0, 1], index=int(round(defaults["NVB"])))
            submitted = st.form_submit_button("预测严重不良反应风险", use_container_width=True)

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
        st.info("填写变量后点击预测，即可显示风险概率、分层和指导意见。")
        return

    pred = predict_probabilities(patient)
    prob = float(pred["Final_Ensemble"].iloc[0])
    level = risk_level(prob)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最终集成模型概率", f"{prob:.1%}")
    col2.metric("AdaBoost", f"{float(pred['AdaBoost'].iloc[0]):.1%}")
    col3.metric("SVC", f"{float(pred['SVC'].iloc[0]):.1%}")
    col4.metric("SVM_RBF", f"{float(pred['SVM_RBF'].iloc[0]):.1%}")
    st.progress(min(max(prob, 0.0), 1.0), text=f"风险分层：{level}；内部Youden阈值：{RISK_THRESHOLD:.3f}")

    if risk_color(prob) == "red":
        st.error(f"预测结果：{level}")
    elif risk_color(prob) == "orange":
        st.warning(f"预测结果：{level}")
    else:
        st.success(f"预测结果：{level}")

    st.markdown("### 临床辅助指导意见")
    for item in clinical_guidance(prob):
        st.markdown(f"- {item}")

    st.markdown("### 当前患者局部贡献提示（快速近似）")
    st.caption("说明：这里不是重新计算SHAP，而是将单个变量替换为训练队列参考值后观察预测概率变化，用于在线快速提示。")
    contrib = local_perturbation_contribution(patient)
    st.dataframe(contrib, use_container_width=True, hide_index=True)
    st.bar_chart(contrib.set_index("Feature")["Contribution_Approx"])


def page_global_explanation() -> None:
    st.subheader("全局解释与模型证据")
    st.caption("以下图形为项目中已经生成的高清PDF，部署时随应用一起展示。")

    options = {
        "SHAP全局重要性合并图": ASSET_DIR / "global/SHAP_Global_Importance_Bar_Rose_Beeswarm_Combined.pdf",
        "SHAP数值和百分比合并图": ASSET_DIR / "global/SHAP_combined_with_values_and_percentages.pdf",
        "SHAP蜂巢图": ASSET_DIR / "global/SHAP_Beeswarm_Final_Ensemble.pdf",
        "SZL与ZN解释对比": ASSET_DIR / "global/SZL_ZN_Explanation_Comparison_Combined.pdf",
        "PDP单变量rug图": ASSET_DIR / "pdp/PDP_Reference_Rug_8_Combined_2x4.pdf",
        "高低风险瀑布图": ASSET_DIR / "local/High_Low_Risk_SHAP_Waterfall_Combined.pdf",
        "LIME风格局部解释": ASSET_DIR / "local/LIME_Local_Explanations_3_Cases.pdf",
    }
    selected = st.selectbox("选择要查看的解释图", list(options))
    pdf_download_or_embed(options[selected])


def page_batch_prediction() -> None:
    st.subheader("批量预测与下载")
    template_path = DATA_DIR / "sample_batch_template.csv"
    st.download_button(
        "下载批量预测模板 CSV",
        data=template_path.read_bytes(),
        file_name="sample_batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    uploaded = st.file_uploader("上传CSV文件，需包含8个变量列", type=["csv"])
    if uploaded is None:
        st.info("上传后将输出三个基础模型概率、最终集成概率和风险分层。")
        return

    try:
        df = pd.read_csv(uploaded)
        pred = predict_probabilities(df)
        result = pd.concat([prepare_features(df, reference=load_training_data()[0]).reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "下载预测结果 CSV",
            data=result.to_csv(index=False, encoding="utf-8-sig"),
            file_name="AE3_All_batch_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"批量预测失败：{exc}")


def page_deployment_notes() -> None:
    st.subheader("部署与使用说明")
    st.markdown(
        """
        **本应用用途**：根据 8 个变量预测患者发生 3级及以上严重不良反应（AE3_All）的风险。

        **部署流程**：

        1. 本地进入本文件夹。
        2. 安装依赖：`pip install -r requirements.txt`
        3. 本地运行：`streamlit run app.py`
        4. 确认页面无误后，将本文件夹上传到 GitHub。
        5. 在 Streamlit Community Cloud 选择该仓库，主文件填写 `app.py`。

        **重要提醒**：本模型为临床辅助决策工具，不能替代医生判断；任何治疗调整均需结合患者实际情况、药品说明书、指南和多学科讨论。
        """
    )


def main() -> None:
    inject_style()
    st.title("严重不良反应预测模型")
    st.caption("Final ensemble: AdaBoost + SVC + SVM_RBF | Outcome: AE3_All")

    st.sidebar.header("模型说明")
    st.sidebar.info(
        f"输入 Ki-67、PR (%)、HER2 lines、ER、Lung、HER2 ADC、Viscral、NVB 后，"
        f"系统输出 AE3_All 阳性预测概率。当前高风险阈值采用内部Youden阈值：{RISK_THRESHOLD:.3f}。"
    )
    st.sidebar.markdown(
        """
        **风险分层**

        - 低风险：< 15%
        - 中风险：15% 到内部Youden阈值
        - 高风险：≥ 内部Youden阈值
        """
    )

    tabs = st.tabs(["单患者预测", "全局解释页面", "批量预测与下载", "部署说明"])
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
