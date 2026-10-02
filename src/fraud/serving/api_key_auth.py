"""Autentikasi API key untuk endpoint scoring yang dipanggil sistem otorisasi pembayaran."""

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"
_API_KEY_ENV_VAR = "SCORING_API_KEY"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def load_scoring_api_key() -> str:
    """Membaca API key yang sah dari environment variable `SCORING_API_KEY`.

    Returns:
        API key yang sah.

    Raises:
        RuntimeError: Variabel tidak diset, kosong, atau masih berisi placeholder contoh.
    """
    key = os.environ.get(_API_KEY_ENV_VAR, "")
    # Kunci kosong akan cocok dengan header kosong, jadi harus ditolak sebagai konfigurasi.
    if not key or key.startswith("<"):
        raise RuntimeError(f"{_API_KEY_ENV_VAR} belum diisi dengan kunci yang sah.")
    return key


def verify_api_key(provided_key: str | None = Security(_api_key_header)) -> None:
    """Dependency FastAPI yang menolak request tanpa API key yang sah.

    Args:
        provided_key: Nilai header `X-API-Key`, `None` kalau header tidak dikirim.

    Raises:
        HTTPException: 401 untuk header yang hilang maupun salah, dengan pesan yang sama
            supaya pemanggil tidak bisa membedakan keduanya.
        RuntimeError: API key sah belum dikonfigurasi di server.
    """
    expected_key = load_scoring_api_key()
    # Perbandingan waktu konstan: == biasa berhenti di karakter pertama yang beda, sehingga
    # lama respons membocorkan berapa karakter awal tebakan yang sudah benar.
    if provided_key is None or not secrets.compare_digest(
        provided_key.encode(), expected_key.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
