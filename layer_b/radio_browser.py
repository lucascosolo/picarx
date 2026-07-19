#!/usr/bin/env python3
# layer_b/radio_browser.py
"""
Minimal client for the free radio-browser.info station directory,
following their API guidelines (https://api.radio-browser.info):

  - servers are DISCOVERED via DNS (all.api.radio-browser.info),
    reverse-resolved to hostnames (TLS needs the name, not the IP),
    shuffled, and tried in order - never a single hardcoded server
  - a speaking User-Agent is sent on every request
  - every station actually played is reported via /json/url/<uuid>
    (their "click" endpoint), which is how the directory learns which
    stations are alive and popular

Stdlib only (socket + urllib), fail-soft everywhere: any DNS/network/
JSON problem returns [] / False and the caller degrades to the saved
station list. Server discovery is cached for an hour so a search
costs one HTTP request, not a DNS walk every time.
"""
import json
import random
import socket
import time
import urllib.parse
import urllib.request

USER_AGENT = "picarx-robot/1.0"
DNS_NAME = "all.api.radio-browser.info"
# Used only if DNS discovery yields nothing (e.g. resolver blocks
# reverse lookups) - the docs' own long-lived example server.
FALLBACK_SERVERS = ["de1.api.radio-browser.info"]
SERVER_CACHE_SEC = 3600.0
HTTP_TIMEOUT = 5.0
SEARCH_LIMIT = 10


class RadioBrowser:
    def __init__(self):
        self._servers = []
        self._servers_at = 0.0

    # ---------- server discovery ----------

    def _discover(self):
        names = set()
        try:
            infos = socket.getaddrinfo(DNS_NAME, 443, proto=socket.IPPROTO_TCP)
            for *_ignored, sockaddr in infos:
                ip = sockaddr[0]
                try:
                    names.add(socket.gethostbyaddr(ip)[0])
                except (socket.herror, socket.gaierror, OSError):
                    continue
        except (socket.gaierror, OSError):
            pass
        servers = sorted(names) or list(FALLBACK_SERVERS)
        random.shuffle(servers)
        return servers

    def servers(self):
        now = time.time()
        if not self._servers or now - self._servers_at > SERVER_CACHE_SEC:
            self._servers = self._discover()
            self._servers_at = now
        return self._servers

    # ---------- requests (try each server until one answers) ----------

    def _get(self, path):
        for server in self.servers():
            url = f"https://{server}{path}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    return json.load(resp)
            except Exception as e:
                print(f"RadioBrowser: {server} failed ({e}), trying next")
        return None

    # ---------- public API ----------

    def search(self, keywords, limit=SEARCH_LIMIT):
        """Find live stations matching free-text keywords. Tries a tag
        search first (genres like 'soft rock' are tags there), then a
        name search. Returns [{'name','url','uuid','tags','country'}]."""
        q = urllib.parse.quote((keywords or "").strip().lower())
        if not q:
            return []
        base = (f"&limit={limit}&hidebroken=true&order=votes&reverse=true")
        for path in (f"/json/stations/search?tagList={q}{base}",
                     f"/json/stations/search?name={q}{base}"):
            rows = self._get(path)
            if rows:
                out = []
                for r in rows:
                    url = r.get("url_resolved") or r.get("url")
                    if not url:
                        continue
                    out.append({
                        "name": (r.get("name") or "unnamed station").strip(),
                        "url": url,
                        "uuid": r.get("stationuuid"),
                        "tags": r.get("tags", ""),
                        "country": r.get("countrycode", ""),
                    })
                if out:
                    return out
        return []

    def click(self, station_uuid):
        """Tell the directory we're playing this station (their requested
        popularity/liveness signal). Fire-and-forget; failure is fine."""
        if not station_uuid:
            return False
        return self._get(f"/json/url/{station_uuid}") is not None
