"""HTTP client for Screener.in company Excel exports (session-authenticated)."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from requests import Response
from bs4 import BeautifulSoup

BASE_URL = "https://www.screener.in"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
XLSX_MAGIC = b"PK\x03\x04"


def _normalize_cookie_domain(domain: str) -> str:
    """Map exporter quirks to a domain requests will send to www.screener.in."""
    d = domain.strip().lower()
    if d in (".www.screener.in", "www.screener.in", ".screener.in", "screener.in"):
        return ".screener.in"
    return domain


@dataclass(frozen=True)
class ScreenerCompanyRef:
    nse_symbol: str
    screener_slug: str
    company_url: str
    warehouse_id: str
    company_id: Optional[str] = None
    page_html: Optional[str] = None
    consolidated: bool = True


class ScreenerError(Exception):
    """Base error for Screener.in client."""


class ScreenerAuthError(ScreenerError):
    """Session missing or not logged in."""


class ScreenerNotFoundError(ScreenerError):
    """Symbol not found on Screener.in."""


class ScreenerExportError(ScreenerError):
    """Export request failed or response was not an Excel file."""


class ScreenerRateLimitError(ScreenerError):
    """Screener.in returned HTTP 429 too many times."""


class ScreenerSession:
    """Authenticated requests session for Screener.in."""

    def __init__(
        self,
        *,
        cookies_file: Optional[Path] = None,
        timeout: float = 60.0,
        rate_limit_max_retries: int = 6,
        rate_limit_base_seconds: float = 30.0,
        request_pause_seconds: float = 0.75,
    ) -> None:
        self.timeout = timeout
        self.rate_limit_max_retries = rate_limit_max_retries
        self.rate_limit_base_seconds = rate_limit_base_seconds
        self.request_pause_seconds = request_pause_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if cookies_file is not None:
            self._load_cookies_file(cookies_file)
        else:
            self._load_cookies_from_env()

    def _load_cookies_file(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Cookies file not found: {path}")
        self._load_cookies_netscape(path)

    def _load_cookies_netscape(self, path: Path) -> None:
        """Parse Netscape cookies.txt; fix domains some browser extensions get wrong."""
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                continue
            domain, _flag, cookie_path, _secure, _expires, name, value = parts
            domain = _normalize_cookie_domain(domain)
            self.session.cookies.set(
                name,
                value,
                domain=domain,
                path=cookie_path or "/",
            )

    def _load_cookies_from_env(self) -> None:
        sessionid = os.getenv("SCREENER_SESSIONID", "").strip()
        if not sessionid:
            raise ScreenerAuthError(
                "Set SCREENER_SESSIONID in .env or pass cookies_file in config. "
                "Export browser cookies for screener.in after logging in."
            )
        self.session.cookies.set("sessionid", sessionid, domain=".screener.in")
        csrftoken = os.getenv("SCREENER_CSRFTOKEN", "").strip()
        if csrftoken:
            self.session.cookies.set("csrftoken", csrftoken, domain=".screener.in")

    def verify_logged_in(self) -> str:
        """Return a short label if the session can access logged-in pages."""
        resp = self.session.get(
            f"{BASE_URL}/watchlist/",
            timeout=self.timeout,
            allow_redirects=False,
        )
        location = resp.headers.get("location", "")
        if resp.status_code in (301, 302) and ("/register" in location or "/login" in location):
            raise ScreenerAuthError(
                "Not logged in. Log in at https://www.screener.in in your browser, "
                "then export cookies (sessionid, csrftoken) into .env or a cookies file."
            )
        if resp.status_code != 200:
            raise ScreenerAuthError(
                f"Unexpected response checking session: HTTP {resp.status_code}"
            )
        return "logged in"

    def _wait_for_rate_limit(self, resp: Response, attempt: int) -> None:
        wait = self.rate_limit_base_seconds * (2**attempt)
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                wait = max(wait, float(retry_after))
            except ValueError:
                pass
        print(
            f"    rate limited (429), waiting {wait:.0f}s "
            f"(retry {attempt + 1}/{self.rate_limit_max_retries}) ...",
            flush=True,
        )
        time.sleep(wait)

    def _request(self, method: str, url: str, **kwargs) -> Response:
        last_resp: Optional[Response] = None
        for attempt in range(self.rate_limit_max_retries + 1):
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            last_resp = resp
            if resp.status_code != 429:
                return resp
            if attempt >= self.rate_limit_max_retries:
                break
            self._wait_for_rate_limit(resp, attempt)
        assert last_resp is not None
        raise ScreenerRateLimitError(
            f"HTTP 429 Too Many Requests for {url} after "
            f"{self.rate_limit_max_retries + 1} attempts"
        )

    def _get(self, url: str, **kwargs) -> Response:
        resp = self._request("GET", url, **kwargs)
        if self.request_pause_seconds > 0:
            time.sleep(self.request_pause_seconds)
        return resp

    def _post(self, url: str, **kwargs) -> Response:
        return self._request("POST", url, **kwargs)

    def search_symbol(self, query: str) -> Optional[dict]:
        resp = self._get(
            f"{BASE_URL}/api/company/search/",
            params={"q": query},
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        q = query.upper()
        for item in results:
            url = item.get("url", "")
            slug = url.strip("/").split("/")[-2] if "/company/" in url else ""
            if slug.upper() == q:
                return item
        return results[0]

    def _fetch_company_page(
        self,
        slug: str,
        *,
        consolidated: bool,
    ) -> tuple[str, Response]:
        path = f"/company/{slug}/consolidated/" if consolidated else f"/company/{slug}/"
        company_url = urljoin(BASE_URL, path)
        resp = self._get(company_url, allow_redirects=True)
        return company_url, resp

    def resolve_company(
        self,
        nse_symbol: str,
        *,
        consolidated: bool = True,
    ) -> ScreenerCompanyRef:
        slug = nse_symbol.upper()
        company_url, resp = self._fetch_company_page(slug, consolidated=consolidated)
        if resp.status_code == 404 or "Page not found" in resp.text:
            hit = self.search_symbol(slug)
            if not hit:
                raise ScreenerNotFoundError(f"No Screener.in company for symbol {nse_symbol!r}")
            company_url = urljoin(BASE_URL, hit["url"])
            resp = self._get(company_url, allow_redirects=True)
            slug = company_url.rstrip("/").split("/")[-2]
            consolidated = "/consolidated/" in company_url
        elif resp.status_code >= 400:
            resp.raise_for_status()

        warehouse_id, company_id = _parse_company_ids(resp.text)
        used_consolidated = consolidated
        if not warehouse_id and consolidated:
            # Some companies have no consolidated export; standalone page still works.
            company_url, resp = self._fetch_company_page(slug, consolidated=False)
            if resp.status_code >= 400 and resp.status_code != 404:
                resp.raise_for_status()
            warehouse_id, company_id = _parse_company_ids(resp.text)
            used_consolidated = False

        if not warehouse_id:
            raise ScreenerExportError(
                f"Could not find data-warehouse-id on {company_url} "
                "(export may require a logged-in premium account)."
            )
        return ScreenerCompanyRef(
            nse_symbol=nse_symbol,
            screener_slug=slug,
            company_url=company_url,
            warehouse_id=warehouse_id,
            company_id=company_id,
            page_html=resp.text,
            consolidated=used_consolidated,
        )

    def export_excel(
        self,
        ref: ScreenerCompanyRef,
    ) -> bytes:
        if ref.page_html:
            page_html = ref.page_html
        else:
            page = self._get(ref.company_url)
            page.raise_for_status()
            page_html = page.text
        csrf = _parse_csrf(page_html)
        if not csrf:
            raise ScreenerExportError("Missing CSRF token on company page.")

        export_url = urljoin(BASE_URL, f"/user/company/export/{ref.warehouse_id}/")
        next_path = ref.company_url.replace(BASE_URL, "")
        resp = self._post(
            export_url,
            data={"csrfmiddlewaretoken": csrf, "next": next_path},
            headers={"Referer": ref.company_url},
            allow_redirects=True,
        )

        if resp.status_code in (401, 403):
            raise ScreenerAuthError(
                f"Export denied ({resp.status_code}). "
                "Ensure you are logged in and have Export-to-Excel access on Screener.in."
            )
        if resp.url.rstrip("/").endswith("/register") or "/login/" in resp.url:
            raise ScreenerAuthError("Export redirected to login — refresh your session cookies.")

        content = resp.content
        if not content.startswith(XLSX_MAGIC):
            snippet = content[:200].decode("utf-8", errors="replace")
            raise ScreenerExportError(
                f"Expected .xlsx from {export_url}, got {resp.status_code} "
                f"({resp.headers.get('Content-Type', 'unknown')}): {snippet[:120]!r}"
            )
        return content


def _parse_company_ids(html: str) -> tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("[data-warehouse-id]")
    warehouse_id = node.get("data-warehouse-id") if node else None
    company_id = node.get("data-company-id") if node else None
    if warehouse_id:
        return str(warehouse_id), str(company_id) if company_id else None
    match = re.search(r'formaction="/user/company/export/(\d+)/"', html)
    if match:
        return match.group(1), None
    return None, None


def _parse_csrf(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one('input[name="csrfmiddlewaretoken"]')
    if tag and tag.get("value"):
        return str(tag["value"])
    match = re.search(
        r'name="csrfmiddlewaretoken"\s+value="([^"]+)"',
        html,
    )
    return match.group(1) if match else None
