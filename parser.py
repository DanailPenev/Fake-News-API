import urllib.request
from bs4 import BeautifulSoup

def parse(url):
	with urllib.request.urlopen(url) as f:
		soup = BeautifulSoup(f, 'html.parser')
		return soup.title.string