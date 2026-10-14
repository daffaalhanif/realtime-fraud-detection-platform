"""Autentikasi dashboard: login dengan password dan token JWT yang membawa peran pengguna.

Ada dua akun, satu per peran, dibaca dari environment variable dengan password berupa hash
scrypt. Token ditandatangani dengan satu kunci rahasia, membawa nama pengguna dan peran, dan
kedaluwarsa tanpa refresh token: setelah itu pengguna login lagi. Halaman yang boleh dilihat
ditentukan dari peran di dalam token yang sudah lolos pemeriksaan, bukan dari pilihan pengguna.

Membuat hash password untuk environment variable:
    uv run python -m fraud.dashboard.auth hash-password
"""

import base64
import getpass
import hashlib
import hmac
import os
import secrets
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import cache

import jwt

# Algoritma HMAC yang boleh dipilih lewat konfigurasi beserta panjang kunci minimalnya, yaitu
# ukuran keluaran hash-nya (RFC 7518). Algoritma lain, termasuk "none", ditolak sejak konfigurasi
# dibaca.
MIN_SECRET_LENGTH = {"HS256": 32, "HS384": 48, "HS512": 64}

MIN_PASSWORD_LENGTH = 12

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
HASH_BYTES = 32
HASH_PREFIX = "scrypt"

INVALID_CREDENTIALS = "Nama pengguna atau password salah."


class Role(StrEnum):
    """Peran pengguna dashboard; masing-masing hanya mengakses halamannya sendiri."""

    ANALYST = "analyst"
    ML_ENGINEER = "ml_engineer"


class AuthError(Exception):
    """Token tidak sah: rusak, kedaluwarsa, kunci tanda tangan berbeda, atau klaim tidak lengkap."""


class AuthConfigError(ValueError):
    """Konfigurasi autentikasi tidak lengkap atau tidak aman."""


@dataclass(frozen=True)
class Claims:
    """Isi token yang sudah lolos pemeriksaan.

    Attributes:
        username: Nama pengguna pemilik token.
        role: Peran pengguna.
    """

    username: str
    role: Role


@dataclass(frozen=True)
class AuthSettings:
    """Konfigurasi autentikasi.

    Attributes:
        secret: Kunci penandatanganan token.
        algorithm: Algoritma penandatanganan, salah satu dari `ALLOWED_ALGORITHMS`.
        expiry_minutes: Masa berlaku token sejak diterbitkan.
        accounts: Nama pengguna ke pasangan hash password dan peran.
    """

    secret: str
    algorithm: str
    expiry_minutes: int
    accounts: dict[str, tuple[str, Role]]


def hash_password(password: str) -> str:
    """Hash scrypt berformat `scrypt$N$r$p$salt$hash`, dengan parameter ikut tersimpan."""
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=HASH_BYTES
    )
    salt_text = base64.b64encode(salt).decode()
    digest_text = base64.b64encode(digest).decode()
    return f"{HASH_PREFIX}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt_text}${digest_text}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Memeriksa password terhadap hash; hash yang formatnya salah dianggap tidak cocok."""
    try:
        prefix, n, r, p, salt_text, digest_text = stored_hash.split("$")
        if prefix != HASH_PREFIX:
            return False
        salt = base64.b64decode(salt_text)
        expected = base64.b64decode(digest_text)
        actual = hashlib.scrypt(
            password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


@cache
def _unknown_user_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


def load_settings() -> AuthSettings:
    """Membaca konfigurasi dari environment variable.

    Raises:
        AuthConfigError: Ada variabel yang kosong, kunci terlalu pendek untuk algoritmanya,
            algoritma tidak diizinkan, masa berlaku tidak valid, atau nama pengguna kedua
            peran sama.
    """
    algorithm = _required("JWT_ALGORITHM")
    if algorithm not in MIN_SECRET_LENGTH:
        allowed = list(MIN_SECRET_LENGTH)
        raise AuthConfigError(f"JWT_ALGORITHM harus salah satu dari {allowed}.")

    secret = _required("JWT_SECRET_KEY")
    if len(secret) < MIN_SECRET_LENGTH[algorithm]:
        raise AuthConfigError(
            f"JWT_SECRET_KEY minimal {MIN_SECRET_LENGTH[algorithm]} karakter untuk {algorithm}."
        )

    try:
        expiry_minutes = int(_required("JWT_EXPIRY_MINUTES"))
    except ValueError as error:
        raise AuthConfigError("JWT_EXPIRY_MINUTES harus berupa bilangan bulat.") from error
    if expiry_minutes <= 0:
        raise AuthConfigError("JWT_EXPIRY_MINUTES harus lebih dari nol.")

    accounts: dict[str, tuple[str, Role]] = {}
    role_prefixes = (
        ("DASHBOARD_ANALYST", Role.ANALYST),
        ("DASHBOARD_ML_ENGINEER", Role.ML_ENGINEER),
    )
    for prefix, role in role_prefixes:
        username = _required(f"{prefix}_USERNAME")
        if username in accounts:
            raise AuthConfigError("Nama pengguna analis dan ML engineer tidak boleh sama.")
        accounts[username] = (_required(f"{prefix}_PASSWORD_HASH"), role)
    return AuthSettings(secret, algorithm, expiry_minutes, accounts)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AuthConfigError(f"Environment variable {name} belum diisi.")
    return value


def authenticate(settings: AuthSettings, username: str, password: str) -> Role | None:
    """Memeriksa nama pengguna dan password.

    Hash selalu dihitung, juga saat nama pengguna tidak dikenal, supaya waktu respons tidak
    membocorkan nama pengguna mana yang ada.

    Returns:
        Peran pengguna kalau cocok, atau None.
    """
    account = settings.accounts.get(username)
    stored_hash, role = account if account else (_unknown_user_hash(), None)
    matched = verify_password(password, stored_hash)
    return role if matched else None


def issue_token(
    settings: AuthSettings, username: str, role: Role, now: datetime | None = None
) -> str:
    """Menerbitkan token bertanda tangan untuk pengguna yang sudah lolos `authenticate`."""
    issued_at = now or datetime.now(UTC)
    claims = {
        "sub": username,
        "role": role.value,
        "iat": issued_at,
        "exp": issued_at + timedelta(minutes=settings.expiry_minutes),
    }
    return jwt.encode(claims, settings.secret, algorithm=settings.algorithm)


def decode_token(settings: AuthSettings, token: str) -> Claims:
    """Memeriksa tanda tangan, masa berlaku, dan kelengkapan klaim token.

    Raises:
        AuthError: Token tidak sah dengan alasan apa pun.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret,
            # Hanya algoritma yang dikonfigurasi, supaya header token tidak bisa menggantinya.
            algorithms=[settings.algorithm],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
        return Claims(username=str(payload["sub"]), role=Role(payload["role"]))
    except (jwt.PyJWTError, ValueError) as error:
        raise AuthError("Token tidak sah.") from error


def _hash_password_command() -> None:
    password = getpass.getpass("Password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password minimal {MIN_PASSWORD_LENGTH} karakter.")
    if getpass.getpass("Ulangi password: ") != password:
        sys.exit("Kedua password tidak sama.")
    print(hash_password(password))


if __name__ == "__main__":
    if sys.argv[1:] != ["hash-password"]:
        sys.exit("Pemakaian: python -m fraud.dashboard.auth hash-password")
    _hash_password_command()
