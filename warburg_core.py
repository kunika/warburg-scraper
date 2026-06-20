"""
warburg_core.py
--------------------
Core logic for scraping the Warburg Institute Iconographic Database.
Imported by both warburg_scraper.py (CLI) and the Colab notebook.

All database-specific knowledge lives here — URL patterns, cookie handling,
slug extraction, IIIF manifest parsing — so fixes and improvements
automatically benefit both pipelines.
"""

import csv
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright


# ----------------------------------------
# Constants
# ----------------------------------------

BASE_URL = "https://iconographic.warburg.sas.ac.uk"

HEADERS = {"User-Agent": "WarburgScraper/1.0 (your@email.ac.uk)"}

REQUEST_DELAY = 1.5  # seconds between HTTP requests

MULTI_VALUE_SEP = " || "  # separator for multi-valued fields in CSV
# In OpenRefine: Edit cells › Split multi-valued cells
# by separator '||'

CSV_FIELDS = [
    "slug",
    "status",
    "query",
    "query_type",
    "scraped_at",
    "object_url",
    "manifest_url",
    "title",
    "nav_date",
    "date",
    "location",
    "iconography",
    "image_urls",
]


# ----------------------------------------
# Playwright: slug collection
# ----------------------------------------


def extract_slugs_from_html(html: str) -> list[str]:
    """
    Extract object slugs from a rendered Warburg results or category page.

    Objects are linked via JavaScript onclick handlers rather than plain
    anchor tags. The pattern in the page's inline script blocks is:
        window.location.href = 'object-wpc-wid-xxxx';

    We extract the slug (the part after 'object-') from every match.

    Note: some objects have 'id-' prefixed slugs (e.g. 'id-321733').
    These are included in the slug list but return 404 when their object
    pages are fetched — this is expected behaviour for this subset of
    records. See warburg database quirks in the documentation.
    """
    pattern = re.compile(r"window\.location\.href\s*=\s*'object-([a-z0-9-]+)'")
    seen = set()
    slugs = []
    for match in pattern.finditer(html):
        slug = match.group(1)
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs


def save_page_html(page, save_path: Path) -> None:
    """
    Save the current rendered page HTML as a named snapshot.

    Serves two purposes:
      - Audit trail: a timestamped record of what the database showed
        at scrape time (the database is updated periodically).
      - Checkpoint: slugs can be re-extracted from saved HTML without
        re-running the browser if Stage 1b fails partway through.

    Files are named by query type and page number, e.g.:
        pages/category_vpc-taxonomy-000007_page1.html
        pages/keyword_prometheus_page2.html
    """
    html = page.content()
    save_path.write_text(html, encoding="utf-8")
    slug_count = len(extract_slugs_from_html(html))
    print(f"    HTML saved → {save_path.name}  ({slug_count} object references found)")


def dismiss_cookie_banner(page) -> bool:
    """
    Dismiss the Warburg cookie consent banner by clicking the 'Continue' button.

    Targets the button precisely using a CSS selector that requires all three:
      - element type:      <button>
      - type attribute:    type="button"
      - onclick attribute: onclick="setCookiePolicy()"

    Then verifies the text content is "Continue" before clicking, so if the
    site ever changes its cookie implementation we get a clear error rather
    than silently clicking the wrong thing.

    Returns True if the banner was found and dismissed, False if not present.

    Warburg quirk: the cookie consent banner is implemented as:
        <button type="button" onclick="setCookiePolicy()">Continue</button>
    The onclick attribute value is specific enough to use as a selector anchor.
    """
    selector = "button[type='button'][onclick='setCookiePolicy()']"

    try:
        btn = page.locator(selector)
        count = btn.count()

        if count == 0:
            print("    No cookie banner detected — continuing.")
            return False

        if count > 1:
            raise RuntimeError(
                f"Cookie banner: expected 1 button matching {selector!r}, "
                f"found {count}. The page structure may have changed."
            )

        actual_text = btn.first.inner_text(timeout=3_000).strip()
        expected_text = "Continue"
        if actual_text != expected_text:
            raise RuntimeError(
                f"Cookie banner: button text is {actual_text!r}, "
                f"expected {expected_text!r}. Refusing to click."
            )

        print(
            f"    Cookie banner found: "
            f"<button onclick='setCookiePolicy()'>{actual_text}</button>"
            f" — clicking..."
        )
        btn.first.click()
        page.wait_for_load_state("networkidle")

        if btn.count() > 0 and btn.first.is_visible(timeout=2_000):
            raise RuntimeError(
                "Cookie banner button is still visible after clicking. "
                "The banner may not have been dismissed."
            )

        print("    Cookie banner dismissed ✓")
        return True

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Cookie banner: unexpected error — {e}") from e


def wait_for_objects(page, timeout: int = 15_000) -> bool:
    """
    Wait until at least one object reference appears in the page source.
    Returns True if found, False if timed out.
    Saves a screenshot on failure to help diagnose the problem.
    """
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
        return len(extract_slugs_from_html(page.content())) > 0
    except PlaywrightTimeout:
        screenshot_path = "playwright_debug.png"
        page.screenshot(path=screenshot_path)
        print(f"\n    ✗ Timed out waiting for page to load.")
        print(f"    Screenshot saved → {screenshot_path}")
        print(f"    Current URL: {page.url}")
        print(f"    Page title:  {page.title()}")
        return False


def navigate_to_next_page(page, next_page_num: int) -> bool:
    """
    Attempt to click the pagination link for `next_page_num`.

    The site's pagination bar shows a fixed window of page numbers
    (e.g. 1 2 3 4 5 … 8) and a 'Next pages' link to reveal more.
    This function handles both cases:
      - If the numbered link is directly visible, click it.
      - If it's hidden behind '…', click 'Next pages' first to
        expose the next range of numbers, then click the target.

    Returns True if navigation succeeded, False if no further pages exist.

    Warburg quirk: pagination is JavaScript-driven and session-based.
    Page numbers cannot be accessed via URL parameters (e.g. ?page=2)
    — they only work within an active browser session. This is why
    Playwright is required rather than simple HTTP requests.
    """
    target = str(next_page_num)

    def click_page_number() -> bool:
        for link in page.locator("a").all():
            try:
                if link.inner_text(timeout=300).strip() == target:
                    link.click()
                    page.wait_for_load_state("networkidle")
                    return True
            except Exception:
                continue
        return False

    if click_page_number():
        return True

    try:
        next_pages = page.get_by_text("Next pages").first
        if next_pages.is_visible(timeout=2_000):
            next_pages.click()
            page.wait_for_load_state("networkidle")
            if click_page_number():
                return True
    except Exception:
        pass

    return False


def collect_all_slugs(page, query_key: str, pages_dir: Path) -> list[str]:
    """
    Collect object slugs from every page of whatever is currently loaded
    in the Playwright browser, saving an HTML snapshot of each page
    and handling pagination automatically.

    Args:
        page:       Playwright page object
        query_key:  Short identifier used in snapshot filenames,
                    e.g. 'category_vpc-taxonomy-000007' or 'keyword_prometheus'
        pages_dir:  Directory to save HTML snapshots into
    """
    all_slugs = []
    seen = set()
    current_page = 1

    while True:
        print(f"    Page {current_page}...", end=" ", flush=True)
        found = wait_for_objects(page)

        if not found:
            print("    Trying extended wait...", end=" ", flush=True)
            page.wait_for_timeout(5_000)

        html_path = pages_dir / f"{query_key}_page{current_page}.html"
        save_page_html(page, html_path)

        page_slugs = extract_slugs_from_html(page.content())
        new_slugs = [s for s in page_slugs if s not in seen]
        seen.update(new_slugs)
        all_slugs.extend(new_slugs)
        print(f"{len(new_slugs)} new objects (running total: {len(all_slugs)})")

        if not new_slugs and not found:
            print("\n    Could not find any object links on this page.")
            print(
                "    If running headless, try again with --visible to see the browser."
            )
            break

        if not navigate_to_next_page(page, current_page + 1):
            print("    No further pages — slug collection complete.")
            break

        current_page += 1
        time.sleep(0.5)

    return all_slugs


def collect_slugs_by_keyword(
    keyword: str, pages_dir: Path, headless: bool = True
) -> list[str]:
    """
    Open the Warburg homepage, submit a keyword search, and collect
    all result slugs across every page of results.

    Warburg quirk: keyword search is session-based. The search term is
    not reflected in the URL after submission — the server holds the
    result set against the browser session. This means:
      - Simple HTTP requests cannot replicate a search
      - The search term disappears from the search box on page 2+
      - Pagination only works within an active browser session
    Playwright maintains the session across pages, solving all three issues.
    """
    query_key = "keyword_" + re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        pw_page = browser.new_page()

        print(f"  Opening homepage...")
        pw_page.goto(f"{BASE_URL}/home")
        pw_page.wait_for_load_state("networkidle")

        dismiss_cookie_banner(pw_page)

        print(f"  Submitting search: '{keyword}'")
        search_box = pw_page.locator("input[type='text']").first
        search_box.click()
        search_box.fill(keyword)
        search_box.press("Enter")
        pw_page.wait_for_load_state("networkidle")

        slugs = collect_all_slugs(pw_page, query_key, pages_dir)
        browser.close()

    return slugs


def collect_slugs_by_category(
    category_id: str, pages_dir: Path, headless: bool = True
) -> list[str]:
    """
    Navigate to a category URL and collect all object slugs across every page.

    Warburg quirk: although category pages have stable URLs (unlike keyword
    searches), pagination is still JavaScript-driven and session-based.
    The ?page=2 parameter has no effect — Playwright is required.
    """
    category_url = f"{BASE_URL}/category/{category_id}"
    query_key = f"category_{category_id}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        pw_page = browser.new_page()

        print(f"  Loading: {category_url}")
        pw_page.goto(category_url)
        pw_page.wait_for_load_state("networkidle")

        dismiss_cookie_banner(pw_page)

        slugs = collect_all_slugs(pw_page, query_key, pages_dir)
        browser.close()

    return slugs


# Async Playwright: slug collection for Colab/Jupyter


async def async_collect_slugs_by_category(category_id: str,
                                           pages_dir: Path) -> list[str]:
    """
    Async version of collect_slugs_by_category for use in Colab/Jupyter
    environments where a sync_playwright() call would conflict with the
    existing event loop.
    """
    from playwright.async_api import async_playwright, TimeoutError as AsyncTimeout

    category_url = f"{BASE_URL}/category/{category_id}"
    query_key    = f"category_{category_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        pw_page = await browser.new_page()

        print(f"  Loading: {category_url}")
        await pw_page.goto(category_url)
        await pw_page.wait_for_load_state("networkidle")

        await async_dismiss_cookie_banner(pw_page)
        slugs = await async_collect_all_slugs(pw_page, query_key, pages_dir)
        await browser.close()

    return slugs


async def async_collect_slugs_by_keyword(keyword: str,
                                          pages_dir: Path) -> list[str]:
    """
    Async version of collect_slugs_by_keyword for use in Colab/Jupyter
    environments where a sync_playwright() call would conflict with the
    existing event loop.
    """
    from playwright.async_api import async_playwright

    query_key = "keyword_" + re.sub(r"[^a-z0-9]+", "_",
                                     keyword.lower()).strip("_")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        pw_page = await browser.new_page()

        print(f"  Opening homepage...")
        await pw_page.goto(f"{BASE_URL}/home")
        await pw_page.wait_for_load_state("networkidle")

        await async_dismiss_cookie_banner(pw_page)

        print(f"  Submitting search: '{keyword}'")
        search_box = pw_page.locator("input[type='text']").first
        await search_box.click()
        await search_box.fill(keyword)
        await search_box.press("Enter")
        await pw_page.wait_for_load_state("networkidle")

        slugs = await async_collect_all_slugs(pw_page, query_key, pages_dir)
        await browser.close()

    return slugs


async def async_dismiss_cookie_banner(page) -> bool:
    """Async version of dismiss_cookie_banner."""
    selector = "button[type='button'][onclick='setCookiePolicy()']"

    try:
        btn   = page.locator(selector)
        count = await btn.count()

        if count == 0:
            print("    No cookie banner detected — continuing.")
            return False

        if count > 1:
            raise RuntimeError(
                f"Cookie banner: expected 1 button matching {selector!r}, "
                f"found {count}. The page structure may have changed."
            )

        actual_text   = await btn.first.inner_text(timeout=3_000)
        actual_text   = actual_text.strip()
        expected_text = "Continue"
        if actual_text != expected_text:
            raise RuntimeError(
                f"Cookie banner: button text is {actual_text!r}, "
                f"expected {expected_text!r}. Refusing to click."
            )

        print(f"    Cookie banner found: "
              f"<button onclick='setCookiePolicy()'>{actual_text}</button>"
              f" — clicking...")
        await btn.first.click()
        await page.wait_for_load_state("networkidle")

        if await btn.count() > 0 and await btn.first.is_visible(timeout=2_000):
            raise RuntimeError(
                "Cookie banner button is still visible after clicking."
            )

        print("    Cookie banner dismissed ✓")
        return True

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Cookie banner: unexpected error — {e}") from e


async def async_wait_for_objects(page, timeout: int = 15_000) -> bool:
    """Async version of wait_for_objects."""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
        return len(extract_slugs_from_html(await page.content())) > 0
    except Exception:
        screenshot_path = "playwright_debug.png"
        await page.screenshot(path=screenshot_path)
        print(f"\n    ✗ Timed out waiting for page to load.")
        print(f"    Screenshot saved → {screenshot_path}")
        print(f"    Current URL: {page.url}")
        print(f"    Page title:  {await page.title()}")
        return False


async def async_navigate_to_next_page(page, next_page_num: int) -> bool:
    """Async version of navigate_to_next_page."""
    target = str(next_page_num)

    async def click_page_number() -> bool:
        for link in await page.locator("a").all():
            try:
                if (await link.inner_text(timeout=300)).strip() == target:
                    await link.click()
                    await page.wait_for_load_state("networkidle")
                    return True
            except Exception:
                continue
        return False

    if await click_page_number():
        return True

    try:
        next_pages = page.get_by_text("Next pages").first
        if await next_pages.is_visible(timeout=2_000):
            await next_pages.click()
            await page.wait_for_load_state("networkidle")
            if await click_page_number():
                return True
    except Exception:
        pass

    return False


async def async_collect_all_slugs(page, query_key: str,
                                   pages_dir: Path) -> list[str]:
    """Async version of collect_all_slugs."""
    all_slugs    = []
    seen         = set()
    current_page = 1

    while True:
        print(f"    Page {current_page}...", end=" ", flush=True)
        found = await async_wait_for_objects(page)

        if not found:
            print("    Trying extended wait...", end=" ", flush=True)
            await page.wait_for_timeout(5_000)

        html      = await page.content()
        html_path = pages_dir / f"{query_key}_page{current_page}.html"
        html_path.write_text(html, encoding="utf-8")
        slug_count = len(extract_slugs_from_html(html))
        print(f"    HTML saved → {html_path.name}  "
              f"({slug_count} object references found)")

        page_slugs = extract_slugs_from_html(html)
        new_slugs  = [s for s in page_slugs if s not in seen]
        seen.update(new_slugs)
        all_slugs.extend(new_slugs)
        print(f"{len(new_slugs)} new objects (running total: {len(all_slugs)})")

        if not new_slugs and not found:
            print("\n    Could not find any object links on this page.")
            break

        if not await async_navigate_to_next_page(page, current_page + 1):
            print("    No further pages — slug collection complete.")
            break

        current_page += 1
        time.sleep(0.5)

    return all_slugs


# ----------------------------------------
# HTTP: IIIF metadata fetching
# ----------------------------------------


def get_manifest_url(slug: str) -> tuple[str | None, str]:
    """
    Fetch the object page for a slug and extract the IIIF manifest URL.

    Returns (manifest_url, status) where status is one of:
      "ok"                       manifest URL found successfully
      "object page fetch failed"  could not retrieve the object page
      "no manifest link"          page loaded but contained no manifest link

    Warburg quirk: some objects have 'id-' prefixed slugs (e.g. 'id-321733').
    These appear in search results but their object pages return HTTP 404.
    Example error:
        org.apache.hc.client5.http.ClientProtocolException:
        HTTP error 404 : Not Found for
        URL https://iconographic.warburg.sas.ac.uk/id-321822
    This is expected — these records are included in the CSV with status
    'object page fetch failed' so they are not silently lost.
    """
    url = f"{BASE_URL}/object-{slug}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"✗  Object page fetch failed: {e}")
        return None, "object page fetch failed"

    soup = BeautifulSoup(r.text, "html.parser")
    link = soup.find("a", href=re.compile(r"manifest\.json"))
    if not link:
        print(f"✗  No manifest link found")
        return None, "no manifest link"

    return urljoin(BASE_URL, link["href"]), "ok"


def parse_manifest(manifest_url: str, slug: str) -> tuple[dict | None, str]:
    """
    Fetch a IIIF Presentation API v3 manifest and return (record, status).

    Status is one of:
      "success"               record parsed and complete
      "manifest fetch failed" could not retrieve or parse the manifest JSON

    Multi-valued fields (iconography paths, image URLs) are joined with
    MULTI_VALUE_SEP (' || ') to match the OpenRefine pipeline output.
    In OpenRefine: Edit cells › Split multi-valued cells › separator '||'
    """
    try:
        r = requests.get(manifest_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        manifest = r.json()
    except Exception as e:
        print(f"✗  Manifest fetch failed: {e}")
        return None, "manifest fetch failed"

    meta = {}
    iconography_paths = []
    for item in manifest.get("metadata", []):
        label = item.get("label", {}).get("none", [""])[0]
        value = item.get("value", {}).get("none", [""])[0]
        if label == "Iconography":
            iconography_paths.append(value)
        else:
            meta[label] = value

    image_urls = []
    for canvas in manifest.get("items", []):
        for anno_page in canvas.get("items", []):
            for annotation in anno_page.get("items", []):
                body = annotation.get("body", {})
                if body.get("type") == "Image" and body.get("id"):
                    image_urls.append(body["id"])

    record = {
        "slug": slug,
        "status": "success",
        "object_url": f"{BASE_URL}/object-{slug}",
        "manifest_url": manifest_url,
        "title": manifest.get("label", {}).get("none", [""])[0],
        "nav_date": manifest.get("navDate", ""),
        "date": meta.get("Date", ""),
        "location": meta.get("Location", ""),
        "iconography": MULTI_VALUE_SEP.join(iconography_paths),
        "image_urls": MULTI_VALUE_SEP.join(image_urls),
    }
    return record, "success"


def build_failure_row(slug: str, status: str, manifest_url: str | None = None) -> dict:
    """
    Build a minimal CSV row for an object whose metadata could not be fetched.
    Ensures every discovered slug appears in the output CSV regardless of
    whether its metadata was successfully retrieved.
    """
    return {
        "slug": slug,
        "status": status,
        "object_url": f"{BASE_URL}/object-{slug}",
        "manifest_url": manifest_url or "",
    }


# ----------------------------------------
# CSV helpers
# ----------------------------------------


def load_completed_slugs(csv_path: Path) -> set[str]:
    """
    Read slugs already present in an existing output CSV.
    Used by the resume mechanism to skip objects already successfully fetched.
    Returns an empty set if the CSV doesn't exist yet.
    """
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["slug"] for row in csv.DictReader(f) if row.get("slug")}


def write_csv_rows(rows: list[dict], csv_path: Path, append: bool = False) -> None:
    """
    Write rows to a CSV file.

    When append=False (default): creates the file with a header row.
    When append=True: appends rows to an existing file without repeating
    the header. Used by fetch_all_metadata for incremental writes so that
    progress survives interruption.
    """
    if not rows:
        return
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------
# Metadata fetching orchestration
# ----------------------------------------


def fetch_all_metadata(
    slugs: list[str], csv_path: Path, query: str, query_type: str
) -> None:
    """
    For every slug: fetch object page → get manifest URL → parse manifest.
    Writes each record to the CSV immediately (incremental) so progress
    survives interruption. Resumes automatically from where a previous
    run left off.

    Args:
        slugs:      list of object slugs to process
        csv_path:   path to the output CSV
        query:      the original search term or category ID (for provenance)
        query_type: 'keyword' or 'category' (for provenance)
    """
    completed = load_completed_slugs(csv_path)
    if completed:
        print(
            f"  Resuming: {len(completed)} slugs already in "
            f"'{csv_path.name}', skipping."
        )

    remaining = [s for s in slugs if s not in completed]
    if not remaining:
        print("  All slugs already fetched — nothing to do.")
        return

    total = len(remaining)
    skipped = len(slugs) - total
    scraped_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"  {total} to fetch"
        + (f"  ({skipped} already completed)" if skipped else "")
        + ".\n"
    )

    for i, slug in enumerate(remaining, 1):
        print(f"  [{i:>{len(str(total))}}/{total}] {slug}", end="  ", flush=True)

        # Common provenance fields added to every row
        provenance = {
            "query": query,
            "query_type": query_type,
            "scraped_at": scraped_at,
        }

        manifest_url, url_status = get_manifest_url(slug)
        if not manifest_url:
            row = {**build_failure_row(slug, url_status), **provenance}
            write_csv_rows([row], csv_path, append=csv_path.exists())
            time.sleep(REQUEST_DELAY)
            continue

        time.sleep(REQUEST_DELAY)

        record, parse_status = parse_manifest(manifest_url, slug)
        if not record:
            row = {**build_failure_row(slug, parse_status, manifest_url), **provenance}
        else:
            row = {**record, **provenance}
            print(f"✓  {record['title'][:60]}")

        write_csv_rows([row], csv_path, append=csv_path.exists())
        time.sleep(REQUEST_DELAY)


# ----------------------------------------
# Image download
# ----------------------------------------


def download_images(input_csv: Path, images_dir: Path, size: str = "full") -> None:
    """
    Download images listed in the 'image_urls' column of a CSV file.

    Args:
        input_csv:  path to the CSV (raw output or OpenRefine export)
        images_dir: directory to save images into
        size:       IIIF size parameter, one of:
                      'full'   — full resolution  (/full/max/0/default.jpg)
                      'large'  — max 1200px       (/full/!1200,1200/0/default.jpg)
                      'medium' — max 800px        (/full/!800,800/0/default.jpg)

    Images are named {slug}_{asset_id}.jpg.
    Safe to re-run — existing files are skipped.
    """
    size_map = {
        "full": "max",
        "large": "!1200,1200",
        "medium": "!800,800",
    }
    if size not in size_map:
        raise ValueError(f"size must be one of: {list(size_map)}")
    iiif_size = size_map[size]

    images_dir.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "success"]

    if not rows:
        print("  No successful records found in CSV.")
        return

    # Flatten all (slug, url) pairs and rewrite size parameter
    all_pairs = []
    for row in rows:
        for url in row.get("image_urls", "").split(MULTI_VALUE_SEP):
            url = url.strip()
            if url:
                # Replace the size segment in the IIIF URL
                adjusted = re.sub(
                    r"/full/[^/]+/0/default\.jpg",
                    f"/full/{iiif_size}/0/default.jpg",
                    url,
                )
                all_pairs.append((row["slug"], adjusted))

    print(f"\n  {len(all_pairs)} image(s) across {len(rows)} object(s).")
    print(f"  Size: {size}  ({iiif_size})")
    print(f"  Saving to: {images_dir.resolve()}\n")

    downloaded = skipped = failed = 0

    for slug, url in all_pairs:
        match = re.search(r"/iiif/3/(\w+)/", url)
        asset_id = match.group(1) if match else re.sub(r"\W", "_", url[-20:])
        filename = images_dir / f"{slug}_{asset_id}.jpg"

        if filename.exists():
            skipped += 1
            continue

        try:
            r = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            r.raise_for_status()
            with open(filename, "wb") as img_f:
                for chunk in r.iter_content(chunk_size=8192):
                    img_f.write(chunk)
            print(f"  ✓  {filename.name}")
            downloaded += 1
        except requests.RequestException as e:
            print(f"  ✗  {filename.name}: {e}")
            failed += 1

        time.sleep(1.0)

    print(f"\n  {downloaded} downloaded   {skipped} already existed   {failed} failed")
