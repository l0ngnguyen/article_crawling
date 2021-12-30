from datetime import date

import luigi

from logics import ArticleLogics
from utils import get_logger, get_output_paths

PATHS = get_output_paths()
logobj = get_logger(__name__, "logs/all_tasks.log")


class BaseTask(luigi.Task):
    date = luigi.DateParameter(default=date.today())

    def run(self):
        cls_name = self.__class__.__name__
        logobj.info("Start %s. [%s]", cls_name, self.date)

        try:
            self.execute()
        except Exception as e:
            logobj.error("Exception [%s]", e)
            raise

        logobj.info("End %s", cls_name)

    def execute(self):
        pass


class FetchData(BaseTask):
    output_path = PATHS["raw_data"]

    def output(self):
        return luigi.LocalTarget(path=self.output_path)

    def execute(self):
        obj = ArticleLogics(logobj)
        obj.fetch_data_from_db(output_path=self.output().path)


class PreprocessData(BaseTask):
    output_path = PATHS["clean_data"]

    def requires(self):
        return FetchData()

    def output(self):
        return luigi.LocalTarget(path=self.output_path)

    def execute(self):
        obj = ArticleLogics(logobj)
        obj.preprocess_data(
            input_path=self.input().path,
            output_path=self.output().path
        )


class Transform(BaseTask):
    X_matrix_path = PATHS["X_matrix"]
    labels_path = PATHS["labels"]
    vectorizer_path = PATHS["vectorizer"]

    def requires(self):
        return PreprocessData()

    def output(self):
        return {
            "X_matrix": luigi.LocalTarget(path=self.X_matrix_path),
            "labels": luigi.LocalTarget(path=self.labels_path),
            "vectorizer": luigi.LocalTarget(path=self.vectorizer_path),
        }

    def execute(self):
        obj = ArticleLogics(logobj)
        obj.transform_data(
            input_path=self.input().path,
            labels_path=self.labels_path,
            vectorizer_path=self.vectorizer_path,
            X_matrix_path=self.X_matrix_path,
        )


class BuildModel(BaseTask):
    model_path = PATHS["model"]

    def requires(self):
        return Transform()

    def output(self):
        return luigi.LocalTarget(path=self.model_path)

    def execute(self):
        input = self.input()
        obj = ArticleLogics(logobj)
        obj.build_model(
            X_matrix_path=input["X_matrix"].path,
            labels_path=input["labels"].path,
            model_path=self.model_path,
        )


class Recommend(BaseTask):
    input_path = luigi.Parameter("data/documents/")
    output_path = PATHS["recommend"]

    def requires(self):
        return [BuildModel(), Transform()]

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def execute(self):
        targets = self.input()
        obj = ArticleLogics(logobj)
        obj.recommend(
            input_path=self.input_path,
            output_path=self.output_path,
            model_path=targets[0].path,
            vectorizer_path=targets[1]["vectorizer"].path,
            X_matrix_path=targets[1]["X_matrix"].path,
            labels_path=targets[1]["labels"].path,
        )


class ExportDB(BaseTask):
    def requires(self):
        return Recommend()

    def execute(self):
        obj = ArticleLogics(logobj)
        obj.export_recommends_to_db(self.input().path)
