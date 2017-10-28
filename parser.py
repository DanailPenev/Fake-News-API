import urllib.request
from bs4 import BeautifulSoup

def parse(url):
	with urllib.request.urlopen(url) as f:
		soup = BeautifulSoup(f, 'html.parser')
		tld = url.split('/')[2]
		if tld[0:4] == "www.":
			tld = tld[4:]
		return (url,soup.title.string)