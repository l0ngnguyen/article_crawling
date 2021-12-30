# Scrapy settings for web_crawling project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "web_crawling"

SPIDER_MODULES = ["web_crawling.spiders"]
NEWSPIDER_MODULE = "web_crawling.spiders"

ROBOTSTXT_OBEY = True

# Encode export file
FEED_EXPORT_ENCODING = "utf-8"

ITEM_PIPELINES = {
    "web_crawling.pipelines.PreProcessPipeline": 300,
    "web_crawling.pipelines.DBInsertingPipeline": 600,
}
