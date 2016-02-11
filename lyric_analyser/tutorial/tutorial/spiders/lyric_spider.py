from __future__ import print_function
import scrapy
import re


from tutorial.items import LyricItem

BASE_URL = "http://search.letssingit.com/cgi-exe/am.cgi?a=search&artist_id=&l=archive&s="

class LyricSpider(scrapy.Spider):
    name = 'lyricSpider'
    artist = ''
    song = ''

    def __init__(self, *args, **kwargs):
        super(LyricSpider, self).__init__(*args, **kwargs)
        inp = [kwargs.get('query')]
        artist, song = inp[0].split("|")
        artist = artist.split()
        song = song.split()

        url = BASE_URL
        query = "+".join(artist) + "+" + "+".join(song)
        self.start_urls = [url + query]
        self.artist = artist
        self.song = song

    def parse(self, response):
        reg = ".*".join(self.artist) + ".*lyrics.*" + ".*".join(self.song)
        url = response.xpath('//a[re:test(@href, "'+reg+'")]//@href').extract()[0]
        yield scrapy.Request(url, callback=self.parse_lyrics)
        """
        from scrapy.shell import inspect_response
        inspect_response(response, self)
        """

    def parse_lyrics(self, response):
        lyrics = response.xpath('//div[re:test(@id, "lyrics")]//text()').extract()
        for line in lyrics:
            print(line, end='')

    def parse_question(self, response):
        yield {
            'title': response.css('h1 a::text').extract()[0],
            'votes': response.css('.question .vote-count-post::text').extract()[0],
            'body': response.css('.question .post-text').extract()[0],
            'tags': response.css('.question .post-tag::text').extract(),
            'link': response.url,
        }
