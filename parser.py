import urllib.request, requests
from bs4 import BeautifulSoup
import time

def parse(url):
	if 'l.facebook.com/l.php' in url:
		r = requests.get(url)
		h = r.headers
		url = h['Refresh'].split('=')[1]
	with urllib.request.urlopen(url) as f:
		soup = BeautifulSoup(f, 'html.parser')
		return soup.title.string