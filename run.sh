#!/bin/bash#
notify-send -i face-wink "Hello! Spider are running"
.venv/bin/scrapy crawl tuoitre #-O logs/crontab_scheduler_crawled.json
