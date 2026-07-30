# 严重不良反应（Severe Adverse Effect）预测模型 Streamlit 部署说明

本文件夹是可部署的 Streamlit 应用包，用于预测患者发生 3级及以上严重不良反应（Severe Adverse Effect，AE3_All）的风险。

## 1. 本地运行

在终端进入本文件夹：

```bash
cd "/Users/mac/Desktop/test/codex/16.8个变量的shap+streamlit/4.SHAP+Streamlit（）/4.Streamlit应用（√）"
pip install -r requirements.txt
streamlit run app.py
```

## 2. 应用内容

- 单患者预测：输入 8 个变量，输出 AdaBoost、SVC、SVM_RBF 和最终集成模型预测概率。
- 风险分层：使用项目内模型评估得到的内部 Youden 阈值 `0.205443` 作为高风险提示阈值。
- 临床辅助指导意见：根据低/中/高风险输出对应监测和管理建议。
- 全局解释页面：展示已完成的 SHAP、PDP、SZL/ZN 解释对比和代表个体解释图。
- 批量预测：上传 CSV 后批量输出概率、风险分层，并下载结果。

## 3. 上传 GitHub

推荐只上传本文件夹作为一个独立仓库，至少需要包含：

```text
app.py
requirements.txt
runtime.txt
data/
assets/
README_Streamlit部署说明.md
```

## 4. Streamlit Community Cloud 部署

1. 打开 Streamlit Community Cloud。
2. 选择 GitHub 仓库。
3. Main file path 填写：`app.py`
4. Python版本由 `runtime.txt` 固定为 Python 3.11；依赖由 `requirements.txt` 自动安装。
5. 部署完成后，打开页面测试单患者预测和批量预测。

## 5. 注意事项

- 本应用为临床辅助决策工具，不能替代医生判断。
- 当前模型由 Python 在应用启动时根据随包训练数据重建 AdaBoost + SVC + SVM_RBF 集成模型。
- 若后续需要和 RDS 模型完全一致，可将 R 端模型导出为可部署服务，或在 Python 中保存经逐例核对后的模型对象。
- 若上传整个大项目到 GitHub，路径较深且文件较多；为了部署稳定，建议只上传本 Streamlit 文件夹。
- 如果云端提示 `ModuleNotFoundError: sklearn`，请确认 `requirements.txt` 中包含 `scikit-learn==1.5.2`，并确认 `runtime.txt` 和 `app.py` 在同一层文件夹。
