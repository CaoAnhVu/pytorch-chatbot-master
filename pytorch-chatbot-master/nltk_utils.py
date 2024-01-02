import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Chia câu thành mảng từ / mã thông báo
    Mã thông báo có thể là một từ hoặc dấu câu, ký tự hoặc số
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = Tìm dạng gốc của từ
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 cho mỗi từ đã biết tồn tại trong câu, 0 nếu không tồn tại
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Thân từng chữ
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Khởi tạo túi với 0 cho mỗi từ
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag