import re
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")  # NLTK bản mới cần thêm cái này
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Khởi tạo 1 lần, tái sử dụng cho toàn bộ dataset
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str, tag_numbers: bool = False) -> str:
    """Làm sạch text. Nếu tag_numbers=True, thay URL/email/số bằng token đặc biệt thay vì xóa."""
    text = text.lower()

    text = re.sub(r"http\S+|www\.\S+", " urltoken " if tag_numbers else " ", text)
    text = re.sub(r"\S+@\S+", " emailtoken " if tag_numbers else " ", text)

    if tag_numbers:
        text = re.sub(r"[£$€]\s?\d+", " moneytoken ", text)
        text = re.sub(r"\b\d{7,}\b", " phonetoken ", text)
        text = re.sub(r"\b\d{3,6}\b", " shortcodetoken ", text)

    text = re.sub(r"\d+", " ", text)  # số còn lại (nếu có) thì xóa
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text



def tokenize_and_filter(text: str) -> list[str]:
    """Tách từ, loại bỏ stopwords và các token quá ngắn (rác)."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    """Đưa từ về dạng gốc (running -> run)."""
    return [STEMMER.stem(t) for t in tokens]


def preprocess(text: str, use_stemming: bool = True, tag_numbers: bool = False) -> str:
    cleaned = clean_text(text, tag_numbers=tag_numbers)
    tokens = tokenize_and_filter(cleaned)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)