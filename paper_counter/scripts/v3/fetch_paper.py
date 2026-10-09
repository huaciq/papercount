"""Read author-public page links. No authentication or paywall bypass."""
from pathlib import Path
import json
import requests
from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'resources/v3_paper'
out.mkdir(parents=True, exist_ok=True)
url = 'https://www.researchgate.net/publication/273296062_An_Apparatus_and_Method_for_Real-Time_Stacked_Sheets_Counting_With_Line-Scan_Cameras'
try:
    r = requests.get(url, timeout=25)
    data = dict(url=url, status=r.status_code)
    if r.ok:
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a.get('href') for a in soup.find_all('a', href=True) if '.pdf' in a['href'].lower() or 'fulltext' in a['href'].lower()]
        data['public_pdf_links'] = list(dict.fromkeys(links))
        (out / 'author_page.html').write_text(r.text, encoding='utf-8')
    (out / 'retrieval.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data))
except requests.RequestException as e:
    (out / 'retrieval.json').write_text(json.dumps(dict(url=url, error=str(e)), indent=2), encoding='utf-8')
    print(type(e).__name__)
