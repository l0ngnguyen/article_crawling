import pickle

import regex as re
from pyvi.ViTokenizer import tokenize
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils import load_yaml, timing

with open(load_yaml("stopwords"), "r") as f:
    stopwords = f.read().split("\n")


class NLP(object):

    uniChars = (
        "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
        "ùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊ"
        "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    )

    vowel_table = [
        ["a", "à", "á", "ả", "ã", "ạ", "a"],
        ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
        ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
        ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
        ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
        ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
        ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
        ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
        ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
        ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
        ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
        ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
    ]

    def __init__(self) -> None:
        self.dicchar = self._loaddicchar()
        self.vowel_to_ids = self._vowel_to_ids()

    def _loaddicchar(self):
        dic = {}
        char1252 = (
            "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|"
            "ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|"
            "ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|"
            "Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|"
            "Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|"
            "Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
        ).split("|")
        charutf8 = (
            "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|"
            "ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|"
            "ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|"
            "Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|"
            "Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|"
            "Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
        ).split("|")
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    def _vowel_to_ids(self):
        vowel_to_ids = {}

        for i in range(len(self.vowel_table)):
            for j in range(len(self.vowel_table[i]) - 1):
                vowel_to_ids[self.vowel_table[i][j]] = (i, j)

        return vowel_to_ids

    def convert_unicode(self, txt):
        return re.sub(
            r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|"
            r"ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|"
            r"ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|"
            r"Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|"
            r"Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|"
            r"Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
            lambda x: self.dicchar[x.group()],
            txt,
        )

    def chuan_hoa_dau_tu_tieng_viet(self, word):
        if not self.is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        vowel_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.vowel_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == "q":
                    chars[index] = "u"
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == "g":
                    chars[index] = "i"
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.vowel_table[x][0]
            if not qu_or_gi or index != 1:
                vowel_index.append(index)
        if len(vowel_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.vowel_to_ids.get(chars[1])
                    chars[1] = self.vowel_table[x][dau_cau]
                else:
                    x, y = self.vowel_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.vowel_table[x][dau_cau]
                    else:
                        chars[1] = (
                            self.vowel_table[5][dau_cau]
                            if chars[1] == "i"
                            else self.vowel_table[9][dau_cau]
                        )
                return "".join(chars)
            return word

        for index in vowel_index:
            x, y = self.vowel_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.vowel_table[x][dau_cau]
                return "".join(chars)

        if len(vowel_index) == 2:
            if vowel_index[-1] == len(chars) - 1:
                x, y = self.vowel_to_ids[chars[vowel_index[0]]]
                chars[vowel_index[0]] = self.vowel_table[x][dau_cau]
            else:
                x, y = self.vowel_to_ids[chars[vowel_index[1]]]
                chars[vowel_index[1]] = self.vowel_table[x][dau_cau]
        else:
            x, y = self.vowel_to_ids[chars[vowel_index[1]]]
            chars[vowel_index[1]] = self.vowel_table[x][dau_cau]
        return "".join(chars)

    def is_valid_vietnam_word(self, word):
        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, y = self.vowel_to_ids.get(char, (-1, -1))
            if x != -1:
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return False
                    vowel_index = index
        return True

    def chuan_hoa_dau_cau_tieng_viet(self, sentence):
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(
                r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1/\2/\3", word
            ).split("/")
            if len(cw) == 3:
                cw[1] = self.chuan_hoa_dau_tu_tieng_viet(cw[1])
            words[index] = "".join(cw)
        return " ".join(words)

    def remove_html(self, txt):
        return re.sub(r"<[^>]*>", "", txt)

    def preprocess_text(self, document):
        document = self.remove_html(document)
        document = self.convert_unicode(document)
        document = self.chuan_hoa_dau_cau_tieng_viet(document)
        document = tokenize(document)
        document = document.lower()
        # xóa các ký tự không cần thiết
        document = re.sub(
            r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờở"
            r"ỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]",
            repl=" ",
            string=document,
        )
        # xóa khoảng trắng thừa
        document = re.sub(r"\s+", " ", document).strip()
        return document


class FeatureExtraction(object):
    def __init__(self, data) -> None:
        self.data = data
        self.__fit()

    @timing
    def __fit(self):
        print("==>Start learning vectorizer")
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            max_features=50000,
            stop_words=stopwords
        )
        self.vectorizer.fit(self.data)

    @timing
    def transform(self, data):
        print("==>Start transforming matrix")
        return self.vectorizer.transform(data)

    def save_vectorizer(self, path):
        print("==>Saving vectorizer")
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)


class Classifier(object):
    def __init__(self, model=None, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer

    @timing
    def fit(
        self,
        X_data,
        y_data,
        train_size=0.8,
        model=LinearSVC(random_state=0)
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            train_size=train_size,
            random_state=32,
            stratify=y_data,
        )
        self.model = model
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test)

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def evaluate(self, X_test, y_test):
        y_true, y_pred = y_test, self.model.predict(X_test)
        print(classification_report(y_true, y_pred))

    def encode_text(self, text):
        clean_text = NLP().preprocess_text(text)
        return self.vectorizer.transform([clean_text])

    def predict(self, texts):
        if not sparse.issparse(texts[0]):
            texts = [self.encode_text(text) for text in texts]
        return self.model.predict(texts)
