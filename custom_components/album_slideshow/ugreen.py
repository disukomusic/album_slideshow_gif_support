"""UGREEN NAS (UGOS Photos) client and pure parsing helpers.

Talks to the UGOS Photos web app's undocumented API, reverse-engineered by
capturing and replaying its own browser network traffic. There is no
official UGREEN documentation for any of this, and it could change without
notice.

Login flow (mirrors the web app's own login):

1. ``POST /ugreen/v1/verify/check`` with ``{"username": ...}``. The
   ``x-rsa-token`` response header is a base64 PEM RSA public key, used only
   to encrypt the password for step 2.
2. RSA (PKCS#1 v1.5) encrypt the password with that key, using the web app's
   own "encryptLong" scheme (see :func:`_rsa_encrypt_long`).
3. ``POST /ugreen/v1/verify/login`` with the encrypted password. The
   response's ``data`` carries a short plaintext ``token``, a ``static_token``
   (reused verbatim as the ``ugk`` query param on every image URL), a second
   RSA public key (``public_key``, used to encrypt ``token`` for subsequent
   requests), and the account id (``uid``, sent as the ``token_uid`` cookie).
4. Every authenticated call sends a ``token_uid``/``token`` cookie, an
   ``x-ugreen-token`` header (a fresh RSA encryption of the plain token - safe
   to redo per request since PKCS#1 v1.5 padding is randomized), and an
   ``x-ugreen-security-key: md5(plain token)`` header; swapping in a
   different security key breaks an otherwise-valid token.
5. Image bytes come from ``/ugreen/v5/photo/picture/stream`` with just the
   session cookie plus ``ugk=<static_token>`` - no token/security-key headers
   there, likely because a plain ``<img>`` tag can't send custom headers.

Re-authenticating on every coordinator refresh (like ``synology.py`` does)
rather than trying to persist/renew a session keeps this resilient to that.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any
from urllib.parse import urlencode

import async_timeout
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_TIMEOUT = 30
_PAGE_SIZE = 1000
_MAX_ASSETS = 20_000

# The web app always splits into 117-character chunks before RSA-encrypting,
# regardless of actual key size (a 2048-bit key could fit more per chunk).
_RSA_CHUNK_SIZE = 117

_CHECK_PATH = "/ugreen/v1/verify/check"
_LOGIN_PATH = "/ugreen/v1/verify/login"
_ALBUM_LIST_PATH = "/ugreen/v5/photo/album/list"
_ALBUM_PICTURE_LIST_PATH = "/ugreen/v5/photo/album/picture/list"
_PICTURE_STREAM_PATH = "/ugreen/v5/photo/picture/stream"
_PICTURE_INFO_PATH = "/ugreen/v5/photo/picture/info"

# ``album_type`` values from ``album/list``; only 1 has been verified.
ALBUM_TYPE_REGULAR = 1

# ``size_type`` for ``picture/stream``; 3 is what the web app's album grid uses.
STREAM_SIZE_THUMBNAIL = 3


class UGreenAuthError(Exception):
    """Raised when login fails (bad credentials, unexpected response, ...)."""


class UGreenApiError(Exception):
    """Raised when an authenticated API call returns a non-success code."""


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes from a UGOS base URL."""
    return (url or "").strip().rstrip("/")


def _rsa_encrypt_long(plaintext: str, pem_public_key: bytes) -> str:
    """Replicate the UGOS web app's RSA ``encryptLong`` helper.

    See the module docstring for the exact chunking/encoding scheme.
    """
    public_key = load_pem_public_key(pem_public_key)
    chunks = [
        plaintext[i : i + _RSA_CHUNK_SIZE]
        for i in range(0, len(plaintext), _RSA_CHUNK_SIZE)
    ] or [""]
    ciphertext = b"".join(
        public_key.encrypt(chunk.encode("utf-8"), PKCS1v15()) for chunk in chunks
    )
    return base64.b64encode(ciphertext).decode("ascii")


def build_image_url(
    base_url: str,
    picture_id: Any,
    source_album_uuid: str,
    static_token: str,
    *,
    source_album_type: int = ALBUM_TYPE_REGULAR,
    size_type: int = STREAM_SIZE_THUMBNAIL,
    upload_time: Any = 0,
    client_id: str = "home-assistant-album-slideshow-WEB",
) -> str:
    """Build a ``picture/stream`` URL for a photo id.

    ``static_token`` is reused unchanged for every photo in the session.
    """
    params = {
        "id": picture_id,
        "size_type": size_type,
        "source_album_uuid": source_album_uuid,
        "source_album_type": source_album_type,
        "client_id": client_id,
        "upload_time": upload_time,
        "ugk": static_token,
    }
    return f"{normalize_base_url(base_url)}{_PICTURE_STREAM_PATH}?{urlencode(params)}"


def find_album_by_name(
    albums: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Return the ``album/list`` entry whose ``album_name`` matches ``name``.

    Album names are not guaranteed unique on the NAS; the first match wins,
    same as every other name-based lookup in this integration.
    """
    for album in albums:
        if isinstance(album, dict) and album.get("album_name") == name:
            return album
    return None


def parse_photo_meta(item: dict[str, Any]) -> dict[str, Any]:
    """Extract the metadata surfaced from an ``album/picture/list`` item."""
    out: dict[str, Any] = {}
    taken = item.get("create_time_utc")
    if isinstance(taken, (int, float)) and taken > 0:
        out["captured_at"] = int(taken * 1000)
    size = item.get("size")
    if isinstance(size, int) and size > 0:
        out["byte_size"] = size
    width, height = item.get("width"), item.get("height")
    if isinstance(width, int):
        out["width"] = width
    if isinstance(height, int):
        out["height"] = height
    return out


def _float_or_none(value: Any) -> float | None:
    """Parse a ``picture/info`` numeric-as-string field; blanks are missing.

    UGOS returns ``latitude``/``longitude`` as strings, empty (``""``) when
    the photo has no GPS data rather than omitting the key entirely.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def location_label(info: dict[str, Any]) -> str | None:
    """Build a short ``"City, Country"`` label from a ``picture/info`` response."""
    locality = None
    for key in ("city_name", "town", "district"):
        val = info.get(key)
        if isinstance(val, str) and val.strip():
            locality = val.strip()
            break
    region = info.get("province")
    country = info.get("country")
    parts: list[str] = []
    if locality:
        parts.append(locality)
    elif isinstance(region, str) and region.strip():
        parts.append(region.strip())
    if isinstance(country, str) and country.strip():
        parts.append(country.strip())
    return ", ".join(parts) if parts else None


def parse_picture_location(info: dict[str, Any]) -> dict[str, Any]:
    """Extract GPS coordinates and a location label from ``picture/info``.

    Not available from ``album/picture/list`` - this needs a separate
    per-photo call, done as a background enrichment pass (see
    ``coordinator._enrich_ugreen_item``).
    """
    out: dict[str, Any] = {}
    lat = _float_or_none(info.get("latitude"))
    lng = _float_or_none(info.get("longitude"))
    if (
        lat is not None
        and lng is not None
        and not (abs(lat) < 1e-6 and abs(lng) < 1e-6)
    ):
        out["latitude"] = lat
        out["longitude"] = lng
    loc = location_label(info)
    if loc:
        out["location"] = loc
    return out


class UGreenClient:
    """Thin async wrapper over the undocumented UGOS Photos web API."""

    def __init__(self, hass, url: str, username: str, password: str) -> None:
        self.hass = hass
        self.base_url = normalize_base_url(url)
        self.username = username
        self.password = password
        self._uid: Any = None
        self._plain_token: str | None = None
        self._static_token: str | None = None
        self._security_key: str | None = None
        self._token_public_key: bytes | None = None

    @property
    def static_token(self) -> str | None:
        """The ``ugk`` value to use when building image URLs."""
        return self._static_token

    @property
    def image_headers(self) -> dict[str, str]:
        """Session cookie for image bytes (no token/security-key needed there)."""
        if not self._plain_token:
            return {}
        return {
            "Cookie": f"token_uid={self._uid}; token={self._auth_token_header()}"
        }

    def _session(self):
        return async_get_clientsession(self.hass)

    def _auth_token_header(self) -> str:
        assert self._plain_token and self._token_public_key
        return _rsa_encrypt_long(self._plain_token, self._token_public_key)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-ugreen-token": self._auth_token_header(),
            "x-ugreen-security-key": self._security_key or "",
            "client-id": "home-assistant-album-slideshow-WEB",
        }

    def _auth_cookies(self) -> dict[str, str]:
        return {"token_uid": str(self._uid), "token": self._auth_token_header()}

    async def async_login(self) -> None:
        """Log in and store the session token, static token and key material.

        Raises :class:`UGreenAuthError` on any failure (bad credentials,
        unreachable NAS, or a response shape this client does not
        understand).
        """
        session = self._session()
        try:
            async with async_timeout.timeout(_TIMEOUT):
                async with session.post(
                    f"{self.base_url}{_CHECK_PATH}",
                    json={"username": self.username},
                    ssl=False,
                ) as resp:
                    rsa_token_b64 = resp.headers.get("x-rsa-token")
                    await resp.read()
        except Exception as err:
            raise UGreenAuthError(f"Could not reach UGREEN NAS: {err}") from err
        if not rsa_token_b64:
            raise UGreenAuthError(
                "UGREEN NAS did not return an x-rsa-token header "
                "(unexpected /verify/check response)"
            )
        check_public_key = base64.b64decode(rsa_token_b64)
        encrypted_password = _rsa_encrypt_long(self.password, check_public_key)

        try:
            async with async_timeout.timeout(_TIMEOUT):
                async with session.post(
                    f"{self.base_url}{_LOGIN_PATH}",
                    json={
                        "username": self.username,
                        "password": encrypted_password,
                        "keepalive": True,
                        "otp": True,
                        "is_simple": True,
                    },
                    ssl=False,
                ) as resp:
                    data = await resp.json(content_type=None)
        except Exception as err:
            raise UGreenAuthError(f"UGREEN login request failed: {err}") from err

        if not isinstance(data, dict) or data.get("code") != 200:
            msg = data.get("msg") if isinstance(data, dict) else None
            raise UGreenAuthError(f"UGREEN login failed: {msg or data}")

        payload = data.get("data") or {}
        plain_token = payload.get("token")
        public_key_b64 = payload.get("public_key")
        if not plain_token or not public_key_b64:
            raise UGreenAuthError("Unexpected UGREEN login response shape")

        self._uid = payload.get("uid")
        self._plain_token = plain_token
        self._static_token = payload.get("static_token") or plain_token
        self._token_public_key = base64.b64decode(public_key_b64)
        self._security_key = hashlib.md5(plain_token.encode()).hexdigest()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        session = self._session()
        async with async_timeout.timeout(_TIMEOUT):
            async with session.post(
                f"{self.base_url}{path}",
                json=body,
                headers=self._auth_headers(),
                cookies=self._auth_cookies(),
                ssl=False,
            ) as resp:
                return await resp.json(content_type=None)

    @staticmethod
    def _unwrap(data: Any, action: str) -> dict[str, Any]:
        if not isinstance(data, dict) or data.get("code") != 200:
            msg = data.get("msg") if isinstance(data, dict) else None
            code = data.get("code") if isinstance(data, dict) else None
            raise UGreenApiError(f"UGREEN {action} failed (code {code}): {msg or data}")
        return data.get("data") or {}

    async def async_list_albums(self) -> list[dict[str, Any]]:
        """Return every regular, synced and shared album for this account."""
        body = {
            "sort_by": 4,
            "sort_order": 0,
            "album_select": 1,
            "share_select": [1, 2],
            "album_type": [1, 2, 3],
            "limit": 100,
            "offset": 0,
            "get_show_type": 1,
        }
        data = self._unwrap(await self._post(_ALBUM_LIST_PATH, body), "album/list")
        return data.get("result") or []

    async def async_list_album_pictures(
        self, album_uuid: str, album_type: int = ALBUM_TYPE_REGULAR
    ) -> list[dict[str, Any]]:
        """Return every photo in an album, paginated by offset/limit."""
        items: list[dict[str, Any]] = []
        offset = 0
        while len(items) < _MAX_ASSETS:
            body = {
                "album_uuid": album_uuid,
                "album_type": album_type,
                "limit": _PAGE_SIZE,
                "offset": offset,
                "sort_by": 0,
                "sort_order": 0,
                "type_option": {"all": 1, "image": 0, "video": 0, "gif": 0, "live": 0},
                "provider_option": {
                    "provider_mine": 1,
                    "provider_other": 1,
                    "provider_uids": [],
                },
                "total": 1,
            }
            data = self._unwrap(
                await self._post(_ALBUM_PICTURE_LIST_PATH, body), "album/picture/list"
            )
            batch = data.get("list")
            if not isinstance(batch, list) or not batch:
                break
            items.extend(b for b in batch if isinstance(b, dict))
            if len(batch) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return items

    async def async_get_picture_info(
        self,
        picture_id: Any,
        source_album_uuid: str,
        source_album_type: int = ALBUM_TYPE_REGULAR,
    ) -> dict[str, Any]:
        """Return full per-photo metadata, including GPS/address when present.

        Used only for background enrichment (see
        ``coordinator._enrich_ugreen_item``) since ``album/picture/list``
        does not carry this - it costs one extra request per photo.
        """
        session = self._session()
        params = {
            "picture_id": picture_id,
            "source_album_uuid": source_album_uuid,
            "source_album_type": source_album_type,
        }
        async with async_timeout.timeout(_TIMEOUT):
            async with session.get(
                f"{self.base_url}{_PICTURE_INFO_PATH}",
                params=params,
                headers=self._auth_headers(),
                cookies=self._auth_cookies(),
                ssl=False,
            ) as resp:
                data = await resp.json(content_type=None)
        return self._unwrap(data, "picture/info")
