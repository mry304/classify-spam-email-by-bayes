import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from preprocessing import preprocess  # noqa: E402
from naive_bayes_scratch import NaiveBayesScratch  # noqa: E402,F401  (cần để joblib load được)

THRESHOLD = 0.9


@st.cache_resource
def load_models():
    """Chỉ load một lần, các lần bấm sau dùng lại từ bộ nhớ."""
    pipeline = joblib.load(ROOT / "data" / "best_sklearn_model_v2.pkl")
    scratch = joblib.load(ROOT / "data" / "nb_scratch_tfidf.pkl")
    return pipeline, scratch


st.set_page_config(page_title="Phân loại thư rác", page_icon="📧")
st.title("📧 Phân loại thư rác bằng Naïve Bayes")
st.caption("Mô hình: TF-IDF (unigram + bigram) + Multinomial Naïve Bayes, alpha = 0.05. "
           "Huấn luyện trên SMS tiếng Anh nên chỉ nhập tin nhắn tiếng Anh.")

pipeline, scratch_nb = load_models()
vectorizer = pipeline.named_steps["tfidf"]
sk_nb = pipeline.named_steps["nb"]

engine = st.radio(
    "Chọn mô hình:",
    ["scikit-learn (MultinomialNB)", "Tự cài đặt (NaiveBayesScratch)"],
    horizontal=True,
)
text = st.text_area("Nhập nội dung tin nhắn:", height=150)

if st.button("Phân loại", type="primary"):
    if not text.strip():
        st.warning("Vui lòng nhập nội dung tin nhắn.")
    else:
        cleaned = preprocess(text, tag_numbers=True)
        if not cleaned:
            st.warning("Sau tiền xử lý không còn từ nào để phân loại.")
        else:
            X = vectorizer.transform([cleaned])
            if X.nnz == 0:
                st.info("Không có từ nào nằm trong từ vựng của mô hình nên không thể phân loại.")
            else:
                model = sk_nb if engine.startswith("scikit") else scratch_nb
                p_spam = float(model.predict_proba(X)[0, 1])

                if p_spam >= THRESHOLD:
                    st.error(f"🚨 SPAM (điểm spam: {p_spam:.1%})")
                else:
                    st.success(f"✅ HAM, thư thường (điểm spam: {p_spam:.1%})")
                st.caption(f"Ngưỡng quyết định: {THRESHOLD}. "
                           f"Văn bản sau tiền xử lý: `{cleaned}`")

                # Đóng góp của từng đặc trưng = trọng số TF-IDF x log-ratio
                log_ratio = sk_nb.feature_log_prob_[1] - sk_nb.feature_log_prob_[0]
                coo = X.tocoo()
                df = pd.DataFrame({
                    "đặc trưng": vectorizer.get_feature_names_out()[coo.col],
                    "đóng góp": (coo.data * log_ratio[coo.col]).round(3),
                })
                df["nghiêng về"] = np.where(df["đóng góp"] > 0, "spam", "ham")
                df = df.loc[df["đóng góp"].abs().sort_values(ascending=False).index].head(8)

                st.subheader("Các đặc trưng ảnh hưởng nhiều nhất")
                st.dataframe(df, hide_index=True)
                st.caption("Đóng góp dương đẩy về spam, âm đẩy về ham. "
                           "Ngoài ra còn hệ số prior (spam chỉ chiếm khoảng 13% dữ liệu huấn luyện) "
                           "kéo nhẹ về phía ham. Từ không có trong từ vựng bị bỏ qua.")