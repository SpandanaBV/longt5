import os
import requests
from bs4 import BeautifulSoup


def fetch_to_file(url: str, out_path: str = os.path.join("data", "raw", "sample.txt")) -> None:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Simple extraction of visible text
    text = soup.get_text("\n", strip=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    # Example usage: fetches a page and writes to data/raw/sample.txt
    fetch_to_file("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print("Saved scraped content to data/raw/sample.txt")
