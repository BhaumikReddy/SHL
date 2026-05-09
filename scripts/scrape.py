"""
scripts/scrape.py

Scrapes the SHL product catalog (Individual Test Solutions only) and saves
the results to data/catalog.json.

All data is extracted from the listing pages only — no detail-page visits,
so this completes in under a minute.

Usage:
    python scripts/scrape.py
"""

import json
import os
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Test type codes and their full descriptions
TEST_TYPE_LABELS = {
    "A": "A - Ability & Aptitude",
    "B": "B - Biodata & Situational Judgement",
    "C": "C - Competencies",
    "D": "D - Development & 360",
    "E": "E - Assessment Exercises",
    "K": "K - Knowledge & Skills",
    "P": "P - Personality & Behavior",
    "S": "S - Simulations",
}
VALID_TYPE_CODES = set(TEST_TYPE_LABELS.keys())


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None


def scrape_listing_page(soup: BeautifulSoup) -> list[dict]:
    """
    Parse one catalog listing page and return assessment dicts.

    Table structure (Individual Test Solutions section):
      col 0: assessment name + link
      col 1: Remote Testing indicator (green dot = yes, empty = no)
      col 2: Adaptive/IRT indicator
      col 3: Test type letter (K, P, A, B, S, etc.)
      col 4+: additional attributes (we ignore these)

    The page contains TWO catalog tables (pre-packaged + individual).
    We target the one whose pagination links carry &type=1.
    """
    results = []

    # Find all table wrappers
    wrappers = soup.find_all("div", class_="custom__table-responsive")

    # Pick the table whose nearby pagination has type=1 links
    # If we can't distinguish, just use all /products/product-catalog/view/ links
    target_table = None
    for wrapper in wrappers:
        # Look for pagination links near this table that reference type=1
        # Check the nearest following pagination div
        pag = wrapper.find_next("ul", class_=lambda c: c and "pagination" in (c or ""))
        if pag:
            links = pag.find_all("a", href=True)
            for link in links:
                if "type=1" in link["href"] and "type=2" not in link["href"]:
                    target_table = wrapper
                    break
        if target_table:
            break

    # Fallback: use the second table (Individual Test Solutions is typically second)
    if target_table is None and len(wrappers) >= 2:
        target_table = wrappers[1]
    elif target_table is None and len(wrappers) == 1:
        target_table = wrappers[0]

    if target_table is None:
        print("  [WARN] Could not find catalog table on this page.")
        return results

    table = target_table.find("table")
    if not table:
        return results

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]  # skip header

    for row in rows:
        cols = row.find_all("td")
        if not cols:
            continue

        # Col 0: name + URL
        a_tag = cols[0].find("a")
        if not a_tag:
            continue
        name = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        if not name or not href:
            continue
        url = href if href.startswith("http") else BASE_URL + href

        # Col 3 (index 3): primary test type letter
        test_type = ""
        if len(cols) > 3:
            raw = cols[3].get_text(strip=True).upper()
            if raw in VALID_TYPE_CODES:
                test_type = TEST_TYPE_LABELS[raw]

        # Fallback: scan all cols after index 0 for type letters
        if not test_type:
            for col in cols[1:]:
                raw = col.get_text(strip=True).upper()
                if raw in VALID_TYPE_CODES:
                    test_type = TEST_TYPE_LABELS[raw]
                    break

        results.append({
            "name": name,
            "url": url,
            "description": "",   # not available on listing page; left empty
            "test_type": test_type,
        })

    return results


def get_max_start(soup: BeautifulSoup) -> int:
    """Return the highest start= value found in pagination links for type=1."""
    max_start = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "start=" in href and "type=1" in href:
            try:
                val = int(href.split("start=")[1].split("&")[0])
                max_start = max(max_start, val)
            except (ValueError, IndexError):
                pass
    return max_start


def main():
    print("=" * 60)
    print("SHL Catalog Scraper — Individual Test Solutions")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_items: list[dict] = []
    seen_urls: set[str] = set()
    start = 0
    page_num = 0
    max_start_known = None  # will be set after first page

    while True:
        page_num += 1
        url = f"{CATALOG_URL}?start={start}&type=1"
        print(f"\nPage {page_num} — fetching: {url}")

        soup = fetch_page(url)
        if soup is None:
            print("  Stopping due to fetch error.")
            break

        # Learn the catalog size from pagination on first page
        if max_start_known is None:
            max_start_known = get_max_start(soup)
            if max_start_known:
                total_est = max_start_known + 12
                print(f"  Catalog size: ~{total_est} Individual Test Solutions across {total_est // 12} pages")

        items = scrape_listing_page(soup)
        new_count = 0
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
                new_count += 1

        print(f"  New items: {new_count} | Total so far: {len(all_items)}")

        if new_count == 0:
            print("  No new items — end of catalog.")
            break

        start += 12
        if max_start_known is not None and start > max_start_known:
            print("  Reached last page.")
            break

        time.sleep(0.5)

    print(f"\nTotal unique assessments scraped: {len(all_items)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {os.path.abspath(OUTPUT_PATH)}")
    print("Done!")


if __name__ == "__main__":
    main()