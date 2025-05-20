import sys
import streamlit as st
import numpy as np
import packaging.version
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 将 set_page_config 移到最前面
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# 处理 NumPy 版本兼容性
def patch_numpy_version():
    try:
        import numba
        current_numpy_version = packaging.version.parse(np.__version__)
        if current_numpy_version > packaging.version.parse('2.0'):
            numba.__init__._ensure_critical_deps = lambda: None  # 简化
    except ImportError:
        pass

patch_numpy_version()

# 自定义 CSS 样式
st.markdown("""
<style>
.main-container { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
.stNumberInput > div > div > input { background-color: white; border: 1px solid #4a84c4; border-radius: 5px; }
.stSelectbox > div > div > select { background-color: white; border: 1px solid #4a84c4; border-radius: 5px; }
.stButton > button { background-color: #4a84c4; color: white; border: none; border-radius: 5px; padding: 10px 20px; }
.stMetric > div { background-color: white; border-radius: 10px; padding: 10px; text-align: center; }
.stMetric > div > div { color: #4a84c4; }
</style>
""", unsafe_allow_html=True)

# 加载模型和特征
try:
    model_8mm = joblib.load('GBC_8mm_model.joblib')
    model_30mm = joblib.load('GBC_30mm_model.joblib')
    features_8mm = joblib.load('GBC_8mm_features.joblib')
    features_30mm = joblib.load('GBC_30mm_features.joblib')
except Exception as e:
    st.error(f"加载模型时发生错误: {e}")
    st.stop()

# 创建输入表单
def get_user_input(features, nodule_diameter):
    input_data = {}
    mid = len(features) // 2
    col1_features = features[:mid]
    col2_features = features[mid:]

    col1, col2 = st.columns(2)

    for idx, (feature, col) in enumerate(zip(col1_features, [col1, col2])):
        with col:
            if feature == 'Nodule diameter':
                st.write(f"Nodule Diameter (mm): {nodule_diameter}")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                input_data[feature] = st.number_input(
                    f"Enter {feature}", min_value=0.0, step=0.1, key=f"{feature}_input_{idx}")
            else:
                input_data[feature] = st.selectbox(
                    f"Select {feature}", [0, 1],
                    format_func=lambda x: "Female" if x == 0 else "Male" if feature == 'Gender' else ("No" if x == 0 else "Yes"),
                    key=f"{feature}_input_{idx}")

    return input_data

def predict_and_explain(input_data, model, features):
    input_df = pd.DataFrame([{feature: input_data[feature] for feature in features}])
    
    # 预测
    prediction = model.predict_proba(input_df)
    malignancy_prob = prediction[0][1]

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 结果展示
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown("### Prediction Results")
        st.metric("Malignancy Risk", f"{malignancy_prob:.2%}")

    with result_col2:
        st.markdown("### Risk Interpretation")
        if malignancy_prob < 0.2:
            st.success("Low Risk: Close monitoring recommended")
        elif malignancy_prob < 0.5:
            st.warning("Moderate Risk: Further investigation suggested")
        else:
            st.error("High Risk: Immediate clinical consultation advised")

    # SHAP 可视化
    st.markdown("### Feature Impact Analysis")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values[1], input_df, show=False)
    st.pyplot(fig)

def main():
    st.title("🫁 TEB Lung Nodule Malignancy Risk Predictor")

    with st.sidebar:
        st.header("Model Selection")
        nodule_diameter = st.number_input("Nodule Diameter (mm)", min_value=0.0, max_value=50.0, step=0.1)
        
        if nodule_diameter <= 8:
            st.info("8mm Model Selected")
            features = features_8mm
            model = model_8mm
        elif nodule_diameter <= 30:
            st.info("30mm Model Selected")
            features = features_30mm
            model = model_30mm
        else:
            st.error("Nodule diameter out of predictive range")
            return

    input_data = get_user_input(features, nodule_diameter)

    if st.sidebar.button("Predict Risk"):
        predict_and_explain(input_data, model, features)

if __name__ == "__main__":
    main()