import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def build_pipeline() -> Pipeline:
    """Tạo pipeline gộp vector hóa + model — đảm bảo fit/transform đúng thứ tự."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])


def train_with_grid_search(X_train, y_train) -> GridSearchCV:
    """
    Tìm alpha (Laplace smoothing) tốt nhất bằng cross-validation trên tập Train.
    Không đụng vào X_test ở bước này.
    """
    pipeline = build_pipeline()

    param_grid = {
    "nb__alpha": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,                    # 5-fold cross-validation
        scoring="f1",            # ưu tiên F1 (vì lớp lệch), không phải accuracy
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = joblib.load("../data/train_test_split_v2.pkl")

    grid_search = train_with_grid_search(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV F1-score:", grid_search.best_score_)

    # Lưu model tốt nhất để dùng lại (evaluate.py, Streamlit app...)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "../data/best_sklearn_model_v2.pkl")
    print("Đã lưu model vào data/best_sklearn_model_v2.pkl")