import numpy as np


class NaiveBayesScratch:
    """Multinomial Naive Bayes tự cài đặt, dùng log-probability và Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """X: ma trận đếm từ (n_docs x V), y: nhãn 0/1."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_docs, n_features = X.shape

        self.class_log_prior_ = np.zeros(len(self.classes_))
        self.feature_log_prob_ = np.zeros((len(self.classes_), n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]                                   # các tin thuộc lớp c
            self.class_log_prior_[i] = np.log(X_c.shape[0] / n_docs)

            word_counts = np.asarray(X_c.sum(axis=0)).ravel()  # tổng số lần mỗi từ xuất hiện
            smoothed = word_counts + self.alpha                # Laplace smoothing
            self.feature_log_prob_[i] = np.log(smoothed) - np.log(smoothed.sum())
        return self

    def predict_log_joint(self, X):
        """Điểm log chưa chuẩn hóa của từng lớp: log P(c) + sum_j x_j * log P(w_j|c)."""
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        """Chuẩn hóa về xác suất bằng log-sum-exp (ổn định số học)."""
        log_joint = self.predict_log_joint(X)
        max_ = log_joint.max(axis=1, keepdims=True)
        log_norm = max_ + np.log(np.exp(log_joint - max_).sum(axis=1, keepdims=True))
        return np.exp(log_joint - log_norm)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]