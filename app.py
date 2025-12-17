import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

st.set_page_config(
    page_title="Stroke Prediction - Data Mining",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "src" / "preprocess"
MODEL_DIR = BASE_DIR / "src" / "models"
RESULTS_DIR = BASE_DIR / "data" / "results"

@st.cache_data
def load_logistic_comparison():
    comp_path = RESULTS_DIR / "logistic_comparison.csv"
    comp_df = pd.read_csv(comp_path)

    raw_cm_path = RESULTS_DIR / "confusion_matrix_logistic_raw.csv"
    proc_cm_path = RESULTS_DIR / "confusion_matrix_logistic_processed.csv"

    raw_cm = pd.read_csv(raw_cm_path, index_col=0)
    proc_cm = pd.read_csv(proc_cm_path, index_col=0)

    return comp_df, raw_cm, proc_cm

@st.cache_data
def load_logistic_results():
    raw_path = RESULTS_DIR / "logistic_regression_raw_metrics.csv"
    proc_path = RESULTS_DIR / "logistic_regression_processed_metrics.csv"
    comp_path = RESULTS_DIR / "comparison_logistic_models.csv"

    raw_df = pd.read_csv(raw_path)
    proc_df = pd.read_csv(proc_path)
    comp_df = pd.read_csv(comp_path)

    return raw_df, proc_df, comp_df

@st.cache_data
def load_xgb_results():
    path = RESULTS_DIR / "xgboost_variants_metrics.csv"
    if not path.exists():
        return None
    dfXG = pd.read_csv(path)
    return dfXG

@st.cache_resource
def load_xgb_model():
    return joblib.load(RESULTS_DIR / "xgb.pkl")

@st.cache_resource
def load_lgbm_model():
    model_path = RESULTS_DIR / "lightgbm.pkl"
    return joblib.load(model_path)


@st.cache_data
def load_lgbm_best_info():
    path = RESULTS_DIR / "lightgbm.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_lgbm_confusion():
    path = RESULTS_DIR / "confusion_matrix_lightgbm.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)
    
@st.cache_data
def load_validation_data():
    val_path = DATA_DIR / "validation.csv"
    df_val = pd.read_csv(val_path)
    X_val = df_val.drop(columns=["stroke"])
    y_val = df_val["stroke"]
    return X_val, y_val

st.sidebar.header("📚 Nội dung")
page = st.sidebar.radio(
    "",
    ["1. Giới thiệu và xử lý dữ liệu",
    "2. Mô hình XGBoost",
    "3. Mô hình LightGBM",
    "4. So sánh 2 mô hình"]
)

st.title("Bài tập lớp môn Khai phá dữ liệu: Dự án dự đoán nguy cơ đột quỵ")

if page.startswith("1."):
    st.markdown("### Phần 1 - Giới thiệu, mô tả và xử lý dữ liệu")

    @st.cache_data
    def load_processed_data():
        dfs = {}
        for name, filename in [
            ("train_balanced", "train_balanced.csv"),
            ("train_scaled", "train_scaled.csv"),
            ("validation", "validation.csv"),
            ("test", "test.csv"),
        ]:
            path = DATA_DIR / filename
            if path.exists():
                dfs[name] = pd.read_csv(path)
            else:
                dfs[name] = None
        return dfs

    def load_raw_data():
        data_path = Path("data/raw/healthcare-dataset-stroke-data.csv")
        df = pd.read_csv(data_path)
        return df

    df = load_raw_data()
    dfs = load_processed_data()
    train_balanced = dfs["train_balanced"]
    train_scaled = dfs["train_scaled"]
    val_df = dfs["validation"]
    test_df = dfs["test"]

    data_info_path = DATA_DIR / "data_info.txt"
    feature_names_path = DATA_DIR / "feature_names.txt"

    data_info_text = data_info_path.read_text(encoding="utf-8") if data_info_path.exists() else None
    feature_names = (
        feature_names_path.read_text(encoding="utf-8").splitlines()
        if feature_names_path.exists()
        else None
    )

    st.subheader("1️. Tổng quan thông tin dữ liệu")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Kích thước dữ liệu:**")
        st.write(f"- Số dòng: **{df.shape[0]:,}**")
        st.write(f"- Số cột: **{df.shape[1]}**")

    with col2:
        st.write("**Các cột trong dataset:**")
        st.write(list(df.columns))

    st.markdown("**5 dòng đầu tiên:**")
    st.dataframe(df.head())

    st.subheader("2️. Phân tích missing values")

    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum(),
        "Missing_Percentage": (df.isnull().sum() / len(df) * 100).round(2),
        "Data_Type": df.dtypes.astype(str)
    })

    missing_info = missing_info[missing_info["Missing_Count"] > 0] \
        .sort_values("Missing_Percentage", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Kiểm tra giá trị `'N/A'` dạng số:**")
        if missing_info.empty:
            st.write("✅ Không có cột nào bị thiếu.")
        else:
            st.dataframe(missing_info)

    # Check 'N/A' dạng string trong các cột object
    with col2:
        st.markdown("**Kiểm tra giá trị `'N/A'` dạng string:**")
        na_string_rows = []
        for col in df.columns:
            if df[col].dtype == "object":
                na_count = (df[col] == "N/A").sum()
                if na_count > 0:
                    na_string_rows.append({
                        "Column": col,
                        "N/A_Count": na_count,
                        "N/A_Percentage": na_count / len(df) * 100
                    })

        if len(na_string_rows) == 0:
            st.write("✅ Không có giá trị `'N/A'` dạng string trong các cột category.")
        else:
            na_string_df = pd.DataFrame(na_string_rows)
            na_string_df["N/A_Percentage"] = na_string_df["N/A_Percentage"].round(2)
            st.dataframe(na_string_df)

    st.subheader("3. Phân tích outliers")

    with st.expander("Xem thống kê outlier"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in ["id", "stroke"]:
            if col in numeric_cols:
                numeric_cols.remove(col)

        outlier_rows = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = (df[col] < lower) | (df[col] > upper)
            outlier_count = mask.sum()
            if outlier_count > 0:
                outlier_rows.append({
                    "Column": col,
                    "Outlier_Count": outlier_count,
                    "Outlier_Percentage": outlier_count / len(df) * 100,
                    "Lower_Bound": round(lower, 2),
                    "Upper_Bound": round(upper, 2),
                    "Min": round(df[col].min(), 2),
                    "Max": round(df[col].max(), 2),
                })

        if len(outlier_rows) == 0:
            st.write("✅ Không phát hiện outlier theo IQR cho các cột numeric.")
        else:
            outlier_df = pd.DataFrame(outlier_rows)
            outlier_df["Outlier_Percentage"] = outlier_df["Outlier_Percentage"].round(2)
            st.dataframe(outlier_df)

    st.subheader("4. Kiểm tra dữ liệu trùng lặp và phân phối target")

    duplicates = df.duplicated().sum()
    st.write(f"**Số dòng trùng lặp hoàn toàn:** `{duplicates}`")

    # Phân phối target stroke
    if "stroke" in df.columns:
        stroke_dist = df["stroke"].value_counts().sort_index()
        total = len(df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Bảng phân phối target (`stroke`):**")
            dist_table = pd.DataFrame({
                "stroke": stroke_dist.index,
                "count": stroke_dist.values,
                "percentage": (stroke_dist.values / total * 100).round(2)
            })
            st.dataframe(dist_table)

            if 0 in stroke_dist and 1 in stroke_dist:
                imbalance_ratio = stroke_dist[0] / stroke_dist[1]
                st.write(f"**Imbalance ratio:** ~ `{imbalance_ratio:.2f} : 1`")

        with col2:
            st.markdown("**Biểu đồ phân phối target:**")
            st.bar_chart(stroke_dist)
    else:
        st.error("Không tìm thấy cột `stroke` trong dữ liệu.")

    st.subheader("5. Các biểu đồ phân tích dữ liệu")

    corr_img = FIG_DIR / "correlation_matrix.png"
    dist_img = FIG_DIR / "distribution_analysis.png"
    rel_img = FIG_DIR / "feature_target_relationship.png"

    if dist_img.exists():
        st.markdown("**Phân tích phân phối các biến (distribution analysis):**")
        st.image(dist_img, use_container_width=True)
    else:
        st.warning("Không tìm thấy `distribution_analysis.png`.")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if corr_img.exists():
            st.markdown("**Ma trận tương quan (correlation matrix):**")
            st.image(corr_img, use_container_width=True)
        else:
            st.warning("Không tìm thấy `correlation_matrix.png`.")

    with col2:
        if rel_img.exists():
            st.markdown("**Mối quan hệ giữa feature và target (feature-target relationship):**")
            st.image(rel_img, use_container_width=True)
        else:
            st.warning("Không tìm thấy `feature_target_relationship.png`.")

    st.subheader("6. Tiến hành xử lý tiền dữ liệu")

    st.markdown("""
    - **Làm sạch dữ liệu:**
        - Xử lý giá trị thiếu, đặc biệt ở cột `bmi` (chuyển `'N/A'` → missing, sau đó fill).
        - Loại bỏ/giới hạn các giá trị bất thường (outlier) theo các rule đã thiết kế.
        - Bỏ cột không hữu ích như `id`.

    - **Biến đổi & tạo thêm feature:**
        - Tạo các nhóm tuổi, nhóm BMI, nhóm glucose.
        - Mã hóa các biến phân loại (one-hot encoding) cho các cột như `gender`, `smoking_status`, các cột nhóm, v.v.
        - Thu được tập feature cuối cùng với **26 feature**.

    - **Chia tập & chuẩn hóa:**
        - Chia dữ liệu thành Train / Validation / Test.
        - Chuẩn hóa feature (scaling).
        - Áp dụng **SMOTE** trên tập Train để xử lý mất cân bằng lớp (stroke = 0/1).
    """)

    st.subheader("7. Các tập dữ liệu sau xử lý")

    if train_scaled is not None:
        rows, cols = train_scaled.shape
        st.markdown(f"**Train (scaled):** `({rows}, {cols})`")
        st.dataframe(train_scaled.head())
    else:
        st.warning("Không tìm thấy `train_scaled.csv`.")

    st.write("")

    if train_balanced is not None:
        rows, cols = train_balanced.shape
        st.markdown(f"**Train (balanced sau SMOTE):** `({rows}, {cols})`")
        st.dataframe(train_balanced.head())
    else:
        st.warning("Không tìm thấy `train_balanced.csv`.")

    st.write("")

    if val_df is not None:
        rows, cols = val_df.shape
        st.markdown(f"**Validation:** `({rows}, {cols})`")
        st.dataframe(val_df.head())
    else:
        st.warning("Không tìm thấy `validation.csv`.")

    st.write("")

    if test_df is not None:
        rows, cols = test_df.shape
        st.markdown(f"**Test:** `({rows}, {cols})`")
        st.dataframe(test_df.head())
    else:
        st.warning("Không tìm thấy `test.csv`.")

    st.subheader("8. Ảnh hưởng của việc tiền xử lý dữ liệu")

    comp_df, raw_cm, proc_cm = load_logistic_comparison()

    st.markdown("#### 8.1. Confusion matrix của 2 trường hợp")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RAW data (dữ liệu thô):**")
        fig, ax = plt.subplots()
        sns.heatmap(raw_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix - RAW data")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        st.pyplot(fig)

    with col2:
        st.markdown("**PROCESSED data (dữ liệu đã xử lý):**")
        fig, ax = plt.subplots()
        sns.heatmap(proc_cm, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_title("Confusion Matrix - PROCESSED data")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        st.pyplot(fig)

    st.caption("Hàng = giá trị thực tế, cột = dự đoán. Lớp 1 là bệnh nhân bị đột quỵ.")

    st.markdown("#### 8.2. Bảng so sánh tổng quát")

    comp_display = comp_df.copy()
    comp_display.index = ["Raw data", "Processed data"]

    for col in ["precision", "recall", "f1-score", "accuracy", "time_seconds"]:
        comp_display[col] = comp_display[col].astype(float).round(3)

    st.dataframe(comp_display, use_container_width=True)

    st.caption("""
- **precision / recall / f1-score / accuracy**: metric tổng thể của từng model.
- **time_seconds**: thời gian huấn luyện + dự đoán (xấp xỉ).
""")

elif page.startswith("2."):
    st.markdown("### Phần 2 - Mô hình XGBoost")

    st.subheader("1. Cấu hình mô hình")

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "scale_pos_weight": "pos_weight = (số mẫu lớp 0 / số mẫu lớp 1)",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    st.markdown("""
    - Train trên: **train_balanced.csv** (dữ liệu đã cân bằng bằng SMOTE).
    - Test trên: **test.csv**.
    - Dùng tham số `scale_pos_weight` để xử lý lệch lớp.
    """)
    st.json(xgb_params)

    st.subheader("2. Kết quả các biến thể XGBoost (threshold khác nhau)")

    xgb_results = load_xgb_results()
    if xgb_results is None:
        st.warning(
            "Không tìm thấy `xgboost_variants_metrics.csv`. "
        )
    else:
        st.write("**Bảng metric:**")

        display_cols = ["model", "variant", "threshold",
                        "precision", "recall", "f1-score",
                        "accuracy", "time_seconds"]
        extra_cols = [c for c in ["F2_1"] if c in xgb_results.columns]
        display_cols += extra_cols
        display_cols = [c for c in display_cols if c in xgb_results.columns]

        styled = (
            xgb_results[display_cols]
            .style.format({
                "threshold": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "accuracy": "{:.3f}",
                "time_seconds": "{:.2f}",
                "F2_1": "{:.3f}" if "F2_1" in xgb_results.columns else None,
            })
        )
        st.dataframe(styled, use_container_width=True)
        st.markdown("""
            - **default_0.5**: dùng ngưỡng mặc định 0.5 cho xác suất.
            - **F2_opt**: chọn threshold tối ưu **F2-score** cho lớp 1 (ưu tiên Recall hơn Precision).
            - **Recall_opt**: chọn threshold để **Recall của lớp 1 cao nhất**, với ràng buộc Precision không quá thấp.
        """)
        st.caption("Các metric precision / recall / F1-score được tính cho lớp **1 (stroke = 1)**.")

        # ==== 3. Vẽ biểu đồ so sánh giữa các variant ====
        st.subheader("3. So sánh hiệu năng giữa các biến thể")

        metric_options = ["precision", "recall", "f1-score", "accuracy"]
        if "F2_1" in xgb_results.columns:
            metric_options.insert(0, "F2_1")

        metric_to_plot = st.selectbox("Chọn metric để vẽ:", metric_options)

        plot_df = (
            xgb_results
            .set_index("variant")[metric_to_plot]
            .sort_values(ascending=False)
        )

        st.bar_chart(plot_df)

        # ==== 4. Chọn 1 biến thể để highlight ====
        st.subheader("4. Phân tích chi tiết 1 cấu hình XGBoost")

        variant_names = xgb_results["variant"].unique().tolist()
        chosen_variant = st.selectbox("Chọn biến thể:", variant_names)

        row = xgb_results[xgb_results["variant"] == chosen_variant].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Threshold sử dụng:**", f"`{row['threshold']:.3f}`")
            st.write("**Precision (class 1):**", f"{row['precision']:.3f}")
            st.write("**Recall (class 1):**", f"{row['recall']:.3f}")
            st.write("**F1-score (class 1):**", f"{row['f1-score']:.3f}")

        with col2:
            st.write("**Accuracy:**", f"{row['accuracy']:.3f}")
            st.write("**Thời gian train + đánh giá (s):**", f"{row['time_seconds']:.2f}")
            if "F2_1" in row and not pd.isna(row["F2_1"]):
                st.write("**F2-score (class 1):**", f"{row['F2_1']:.3f}")

elif page.startswith("3."):
    st.markdown("### Phần 3 - Mô hình LightGBM")

    # ===== 1. Cấu hình & tuning =====
    st.subheader("1. Cấu hình & quá trình tuning")

    st.markdown("""
    - Train trên: **train_balanced.csv** (dữ liệu đã cân bằng bằng SMOTE).
    - Validation trên: **validation.csv**.
    - Sử dụng **Grid Search đơn giản trên validation set** với các tham số:
        - `num_leaves`: [15, 31, 63]
        - `max_depth`: [-1, 7, 11]
        - `learning_rate`: [0.01, 0.05, 0.1]
        - `feature_fraction`: [0.8, 1.0]
        - `bagging_fraction`: [0.8, 1.0]
        - `bagging_freq`: [0, 5]
        - `min_child_samples`: [20, 50]
    - Mỗi tổ hợp tham số:
        - Train `LGBMClassifier` với `n_estimators = 5000`, `early_stopping_rounds = 100`.
        - Đo **AUC trên validation** và chọn mô hình tốt nhất.
    """)

    best_info = load_lgbm_best_info()
    if best_info is not None:
        st.markdown("**Các tham số tốt nhất tìm được (trên validation):**")
        st.dataframe(best_info, use_container_width=True)
        if "best_auc_valid" in best_info.columns:
            best_auc_value = best_info["best_auc_valid"].iloc[0]
            st.write(f"**Best AUC (validation):** `{best_auc_value:.3f}`")
    else:
        st.warning("Chưa tìm thấy `lightgbm.csv`")

    st.write("---")

    # ===== 2. Đánh giá lại LightGBM trên validation =====
    st.subheader("2. Hiệu năng LightGBM trên tập validation")

    try:
        lgbm_model = load_lgbm_model()
        X_val, y_val = load_validation_data()
    except Exception as e:
        st.error(f"Không load được model hoặc dữ liệu validation: {e}")
        st.stop()

    # Predict
    y_proba = lgbm_model.predict_proba(X_val)[:, 1]
    y_pred = lgbm_model.predict(X_val)

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_proba)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Accuracy:** `{acc:.3f}`")
        st.write(f"**Precision (class 1):** `{prec:.3f}`")
        st.write(f"**Recall (class 1):** `{rec:.3f}`")
    with col2:
        st.write(f"**F1-score (class 1):** `{f1:.3f}`")
        st.write(f"**AUC (validation):** `{auc:.3f}`")

    st.caption("Các metric precision / recall / F1-score được tính cho lớp **1 (stroke = 1)**.")

    # ===== 3. Confusion matrix =====
    st.subheader("3. Ma trận nhầm lẫn (Confusion Matrix)")

    cm_df = load_lgbm_confusion()
    if cm_df is None:
        # nếu vì lý do gì đó không đọc được file CSV, tự tính lại từ y_val & y_pred
        cm = confusion_matrix(y_val, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"]
        )

    st.markdown("**Bảng confusion matrix:**")
    st.dataframe(cm_df)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(cm_df.values)

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, cm_df.values[i, j],
                    ha="center", va="center")

    ax.set_title("Confusion Matrix - LightGBM (Validation)")
    st.pyplot(fig)

elif page.startswith("4."):
    st.markdown("### Phần 4 - So sánh các mô hình ML")

    st.subheader("1. Bảng so sánh các mô hình")

    try:
        compare_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
        st.dataframe(compare_df, use_container_width=True)
    except Exception as e:
        st.error(f"Không thể load bảng model_comparison.csv: {e}")
        st.stop()

    st.subheader("2. Biểu đồ so sánh các metric")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Precision**")
        st.image(RESULTS_DIR / "precision_plot.png")

        st.markdown("**F1-score**")
        st.image(RESULTS_DIR / "f1_plot.png")

    with col2:
        st.markdown("**Recall**")
        st.image(RESULTS_DIR / "recall_plot.png")

        st.markdown("**Accuracy**")
        st.image(RESULTS_DIR / "accuracy_plot.png")