from sqlalchemy import (Column, DateTime, Integer, String, UnicodeText,
                        create_engine)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import Float

from utils import load_yaml


class ArticleDB:
    def __init__(self) -> None:
        self.connection_info = load_yaml("db-connection-info")
        self.connection_string = (
            f"mysql+pymysql://{self.connection_info['username']}:"
            f"{self.connection_info['password']}"
            f"@{self.connection_info['server']}/"
            f"{self.connection_info['db_name']}"
        )
        self.engine = create_engine(self.connection_string, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_table(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            print("Table create successfully!")
        except Exception as e:
            print(e)
            print("Error while creating table!")

    def create_session(self):
        return self.Session()

    def _insert_obj(self, session, obj):
        try:
            session.add(obj)
            session.commit()
            return obj
        except IntegrityError as e:
            print(e)
            session.rollback()
            return

    def insert_article(self, session, argument):
        article = Article(**argument)
        return self._insert_obj(session, article)

    def insert_tag(self, session, tag_content):
        exist_tag = session.query(Tag).filter_by(tag=tag_content).first()
        if exist_tag is None:
            tag = Tag(tag=tag_content)
            session.add(tag)
            session.commit()
            return tag
        else:
            return exist_tag

    def insert_writer(self, session, writer_name):
        exist_writer = (
            session.query(Writer).filter_by(name=writer_name).first()
        )
        if exist_writer is None:
            writer = Writer(name=writer_name)
            session.add(writer)
            session.commit()
            return writer
        else:
            return exist_writer

    def insert_document(self, session, document):
        new_document = Document(body=document)
        return self._insert_obj(session, new_document)

    def insert_article_recommend(
        self, session, document_id, article_id, similarity
    ):
        ar = ArticleRecommend(
            document_id=document_id,
            article_id=article_id,
            similarity=similarity,
        )
        return self._insert_obj(session, ar)

    def get_the_latest_date(self):
        session = self.create_session()
        query = session.query(func.max(Article.datetime).label("newest_date"))
        newest_date = query.first().newest_date
        session.close()
        return newest_date.date() if newest_date is not None else None

    def get_articles_by_specific_categories(self, session, categories=None):
        if categories is None:
            articles = session.query(
                Article.id,
                Article.category,
                Article.body
            )
        else:
            articles = (
                session.query(Article.id, Article.category, Article.body)
                .filter(Article.category.in_(categories))
                .all()
            )
        return articles

    def get_articles_by_id(self, session, id):
        article = (
            session.query(Article.category, Article.title, Article.body)
            .filter(Article.id == id)
            .first()
        )
        writers = (
            session.query(Writer.name)
            .join(ArticleWriter)
            .filter(ArticleWriter.article_id == id)
            .all()
        )
        tags = (
            session.query(Tag.tag)
            .join(ArticleTag)
            .filter(ArticleTag.article_id == id)
            .all()
        )

        return {
            "category": article[0],
            "title": article[1],
            "body": article[2],
            "writers": [e[0] for e in writers],
            "tags": [e[0] for e in tags],
        }

    def get_recommend_articles_by_document_id(self, session, id):
        recommends = (
            session.query(
                ArticleRecommend.article_id,
                ArticleRecommend.similarity
            )
            .filter(ArticleRecommend.document_id == id)
            .all()
        )

        recommend_articles = []
        for article_id, sim in recommends:
            article = self.get_articles_by_id(session, article_id)
            article["similarity"] = sim
            recommend_articles.append(article)

        return recommend_articles


Base = declarative_base()


class Article(Base):
    __tablename__ = "Article"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    url = Column("url", String(250), nullable=False, unique=True)
    publisher = Column("publisher", String(50))
    datetime = Column("datetime", DateTime)
    title = Column("title", String(250))
    body = Column("body", UnicodeText, nullable=False)
    category = Column("category", String(30), nullable=False)
    writers = relationship(
        argument="Writer", secondary="ArticleWriter", backref="articles"
    )
    tags = relationship(
        argument="Tag", secondary="ArticleTag", backref="articles"
    )


class Writer(Base):
    __tablename__ = "Writer"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    name = Column("name", String(80), nullable=False, unique=True)


class ArticleWriter(Base):
    __tablename__ = "ArticleWriter"
    article_id = Column(
        "article_id", Integer, ForeignKey("Article.id"), primary_key=True
    )
    writer_id = Column(
        "writer_id", Integer, ForeignKey("Writer.id"), primary_key=True
    )


class Tag(Base):
    __tablename__ = "Tag"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    tag = Column("tag", String(80), nullable=False, unique=True)


class ArticleTag(Base):
    __tablename__ = "ArticleTag"
    article_id = Column(
        "article_id", Integer, ForeignKey("Article.id"), primary_key=True
    )
    tag_id = Column("tag_id", Integer, ForeignKey("Tag.id"), primary_key=True)


class Document(Base):
    __tablename__ = "Document"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    body = Column("body", UnicodeText)


class ArticleRecommend(Base):
    __tablename__ = "ArticleRecommend"
    document_id = Column(
        "document_id", Integer, ForeignKey("Document.id"), primary_key=True
    )
    article_id = Column(
        "article_id", Integer, ForeignKey("Article.id"), primary_key=True
    )
    similarity = Column("similarity", Float)


if __name__ == "__main__":
    pass
