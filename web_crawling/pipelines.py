# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


from datetime import datetime

# useful for handling different item types with a single interface
from scrapy.exceptions import DropItem

from models import ArticleDB
from utils import get_logger


class PreProcessPipeline:
    def _convert_datetime(self, text):
        if text is None:
            return None
        text = text.replace("GMT+7", "").strip()
        return datetime.strptime(text, "%d/%m/%Y %H:%M")

    def _get_writers(self, writers):
        if writers is None:
            return None
        return [s.strip() for s in writers.split("-")]

    def process_item(self, item, spider):
        item["writers"] = self._get_writers(item.get("writers"))
        item["datetime"] = self._convert_datetime(item.get("datetime"))
        return item


class DBInsertingPipeline:
    def __init__(self) -> None:
        self.db = ArticleDB()
        self.db.create_table()
        self.logobj = get_logger(__name__, "logs/crawling.log")

    def process_item(self, item, spider):
        session = self.db.create_session()

        # insert new article
        new_article = self.db.insert_article(
            session,
            {
                "url": item["url"],
                "publisher": item["publisher"],
                "datetime": item["datetime"],
                "title": item["title"],
                "body": item["body"],
                "category": item["category"],
            },
        )
        if new_article is None:
            session.close()

            self.logobj.info(f'[CRAWL DATA] Drop article from {item["url"]}')

            raise DropItem()

        # insert tag
        for tag in item["tags"]:
            tag = self.db.insert_tag(session, tag)
            tag.articles.append(new_article)

        # insert writer
        for writer in item["writers"]:
            writer = self.db.insert_writer(session, writer)
            writer.articles.append(new_article)

        session.close()

        self.logobj.info(
            f'[CRAWL DATA] Insert article from {item["url"]} to database'
        )

        return item
