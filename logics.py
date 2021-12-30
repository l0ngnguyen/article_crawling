import json
import os
import pickle

import pandas as pd
from scipy import sparse, spatial
from tqdm import tqdm

from classification import NLP, Classifier, FeatureExtraction
from models import ArticleDB
from utils import load_yaml, timing

tqdm.pandas(desc="Preprocessing!")


class ArticleLogics:
    def __init__(self, logobj) -> None:
        self.db = ArticleDB()
        self.logobj = logobj

    def fetch_data_from_db(self, output_path):
        label_names = load_yaml("labels")
        session = self.db.create_session()

        data = self.db.get_articles_by_specific_categories(
            session, label_names
        )
        df = pd.DataFrame(data, columns=["id", "category", "body"])
        df.to_csv(output_path, index=None)

        self.logobj.info(
            f"[DRY RUN] fetch data from database and save in {output_path}"
        )

        session.close()

    def preprocess_data(self, input_path, output_path):

        df = pd.read_csv(input_path)
        df["body"] = df["body"].astype(str).progress_map(NLP().preprocess_text)

        # drop null values
        df.drop(
            df[df.body.isnull() | df.category.isnull()].index,
            axis=0,
            inplace=True,
        )

        # drop rows that have len(clean_body) < 10
        len_texts = df["body"].astype(str).apply(lambda t: len(t.split()))
        df.drop(len_texts[len_texts <= 10].index, axis=0, inplace=True)

        df.to_csv(output_path, index=False)

        self.logobj.info(
            f"[DRY RUN] preprocess data from {input_path} "
            f"and save in {output_path}"
        )

    def transform_data(
        self,
        input_path,
        vectorizer_path,
        X_matrix_path,
        labels_path
    ):

        data_df = pd.read_csv(input_path)
        data_df[["id", "category"]].to_csv(labels_path, index=False)

        extraction = FeatureExtraction(data_df["body"].astype(str))
        extraction.save_vectorizer(vectorizer_path)
        X_matrix = extraction.transform(data_df["body"])
        sparse.save_npz(X_matrix_path, X_matrix)

        self.logobj.info(
            f"[DRY RUN] transform data from {input_path} and save "
            f"in labels: {labels_path}, vectorizer: {vectorizer_path}"
            f" and X_matrix: {X_matrix_path}"
        )

    def build_model(self, X_matrix_path, labels_path, model_path):

        y_data = pd.read_csv(labels_path, index_col="id").values.ravel()
        X_data = sparse.load_npz(X_matrix_path)

        classifier = Classifier()
        classifier.fit(X_data, y_data)
        classifier.save_model(file_path=model_path)

        self.logobj.info(
            f"[DRY RUN] train model with X_matrix: {X_matrix_path}"
            f"and labels: {labels_path}, and save model in {model_path}"
        )

    def load_classifier_data(
        self,
        model_path,
        vectorizer_path,
        X_matrix_path,
        labels_path
    ):

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        self.classifier = Classifier(model, vectorizer)

        X_matrix = sparse.load_npz(X_matrix_path)
        self.data = pd.read_csv(labels_path, index_col=0)
        self.data["text_encoded"] = pd.Series(
            [X_matrix[i] for i in range(X_matrix.shape[0])],
            index=self.data.index,
        )

    def load_documents(self, data_path):
        documents = []
        file_names = os.listdir(data_path)
        for file_name in file_names:
            path = data_path + file_name
            with open(path, "r") as f:
                documents.append(f.read())
        return documents

    def _get_articles_by_ids(self, ids):
        session = self.db.create_session()
        articles = []
        for id in ids:
            article = self.db.get_articles_by_id(session, id)
            articles.append(article)
        session.close()
        return pd.DataFrame(articles, index=ids)

    def _recommend(self, text, n=10):
        # encode text
        text_encoded = self.classifier.encode_text(text)
        prediction_label = self.classifier.predict(text_encoded)[0]
        text_encoded = text_encoded[0].todense()

        # compute cosine similarity
        articles = self.data[self.data["category"] == prediction_label].copy()
        articles["similarity"] = articles["text_encoded"].apply(
            lambda a: 1 - spatial.distance.cosine(a.todense(), text_encoded)
        )
        articles.sort_values(by=["similarity"], ascending=False, inplace=True)

        headn = articles.head(n)
        top_n = self._get_articles_by_ids(headn.index)
        top_n["similarity"] = headn["similarity"]
        top_n["id"] = headn.index

        return top_n

    @timing
    def recommend(
        self,
        input_path,
        output_path,
        model_path,
        vectorizer_path,
        X_matrix_path,
        labels_path,
    ):
        self.load_classifier_data(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            X_matrix_path=X_matrix_path,
            labels_path=labels_path,
        )

        documents = self.load_documents(input_path)
        recommends = []
        for doc in documents:
            top10 = self._recommend(doc)
            prediction = {
                "text": doc,
                "article_ids": top10["id"].tolist(),
                "similarities": top10["similarity"].tolist(),
            }

            recommends.append(prediction)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(recommends, f, ensure_ascii=False)

        self.logobj.info(
            f"[DRY RUN] Recommend for documents from {input_path}, "
            f"with data for recommendation from {model_path}, "
            f"{labels_path}, {vectorizer_path} and {X_matrix_path}, "
            f"save the result in {output_path}"
        )

    def export_recommends_to_db(self, recommends_path):
        with open(recommends_path, "r") as f:
            recommends = json.load(f)
        session = self.db.create_session()
        for rec in recommends:
            doc_obj = self.db.insert_document(session, rec["text"])
            for id, sim in zip(rec["article_ids"], rec["similarities"]):
                self.db.insert_article_recommend(session, doc_obj.id, id, sim)
        session.close()

        self.logobj.info(
            f"[DRY RUN] Save recommends from {recommends_path} in database"
        )
