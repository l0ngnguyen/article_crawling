from datetime import date, timedelta

import scrapy
from scrapy.loader import ItemLoader

from models import ArticleDB
from utils import get_logger, load_yaml
from web_crawling.items import ArticleItem


class TuoitreSpider(scrapy.Spider):
    name = "tuoitre"

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.categories = load_yaml("categories")
        self.get_crawled_dates()
        self.logobj = get_logger(__name__, "logs/crawling.log")
        self.logobj.info(
            f"[CRAL DATA] Start Crawling Data from "
            f"{self.start_date} to today: {self.end_date}"
        )

    def get_crawled_dates(self):
        db = ArticleDB()
        latest_date = db.get_the_latest_date()
        self.start_date = (
            latest_date if latest_date is not None else date(2017, 1, 1)
        )
        self.end_date = date.today()
        delta = self.end_date - self.start_date

        self.dates = []
        for i in range(delta.days + 1):
            d = self.start_date + timedelta(days=i)
            date_str = f"{d.day}-{d.month}-{d.year}"
            self.dates.append(date_str)

    def start_requests(self):
        for cat in self.categories:
            for d in self.dates:
                url = f"https://tuoitre.vn/{cat}/xem-theo-ngay/{d}.html"
                yield scrapy.Request(url=url, callback=self.parse_links)

    def parse_links(self, response):
        yield from response.follow_all(
            css="h3.title-news a",
            callback=self.parse_details
        )

    def parse_details(self, response):
        item = ArticleItem()
        for field in item.fields:
            item.setdefault(field, None)

        loader = ItemLoader(item, selector=response)
        loader.add_value("url", response.url)
        loader.add_value("publisher", "tuổi trẻ")
        loader.add_css("datetime", "div.date-time::text")
        loader.add_css("title", "h1.article-title::text")
        loader.add_css("body", "#main-detail-body > p")
        loader.add_css(
            "category",
            "div.bread-crumbs.fl > ul > li.fl > a::text"
        )
        loader.add_css("writers", "div.author")
        loader.add_css("tags", "li.tags-item a::text")

        yield loader.load_item()


if __name__ == "__main__":
    pass
