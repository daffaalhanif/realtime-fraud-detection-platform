from datetime import UTC, datetime, timedelta

import jwt
import pytest

from fraud.dashboard import auth
from fraud.dashboard.auth import (
    AuthConfigError,
    AuthError,
    AuthSettings,
    Claims,
    Role,
    authenticate,
    decode_token,
    hash_password,
    issue_token,
    load_settings,
    verify_password,
)

SECRET = "s" * 64
ANALYST_PASSWORD = "analis-password-1"
ENGINEER_PASSWORD = "engineer-password-1"


@pytest.fixture(scope="module")
def settings() -> AuthSettings:
    return AuthSettings(
        secret=SECRET,
        algorithm="HS256",
        expiry_minutes=60,
        accounts={
            "ani": (hash_password(ANALYST_PASSWORD), Role.ANALYST),
            "eko": (hash_password(ENGINEER_PASSWORD), Role.ML_ENGINEER),
        },
    )


def test_password_hash_roundtrip_and_salting() -> None:
    first, second = hash_password("kata-sandi-panjang"), hash_password("kata-sandi-panjang")

    assert verify_password("kata-sandi-panjang", first)
    assert not verify_password("kata-sandi-lain", first)
    assert first != second


@pytest.mark.parametrize("stored", ["", "bukan-hash", "bcrypt$1$2$3$a$b", "scrypt$x$8$1$AA==$AA=="])
def test_malformed_hash_never_matches(stored: str) -> None:
    assert not verify_password("apa-saja", stored)


def test_authenticate_returns_the_role_of_the_matching_account(settings: AuthSettings) -> None:
    assert authenticate(settings, "ani", ANALYST_PASSWORD) is Role.ANALYST
    assert authenticate(settings, "eko", ENGINEER_PASSWORD) is Role.ML_ENGINEER


def test_authenticate_rejects_wrong_password_unknown_user_and_swapped_password(
    settings: AuthSettings,
) -> None:
    assert authenticate(settings, "ani", "salah") is None
    assert authenticate(settings, "tidak-ada", ANALYST_PASSWORD) is None
    assert authenticate(settings, "ani", ENGINEER_PASSWORD) is None


def test_token_roundtrip_carries_username_and_role(settings: AuthSettings) -> None:
    token = issue_token(settings, "ani", Role.ANALYST)

    assert decode_token(settings, token) == Claims("ani", Role.ANALYST)


def test_expired_token_is_rejected(settings: AuthSettings) -> None:
    issued = datetime.now(UTC) - timedelta(minutes=settings.expiry_minutes + 1)

    with pytest.raises(AuthError):
        decode_token(settings, issue_token(settings, "ani", Role.ANALYST, now=issued))


def test_token_signed_with_another_secret_is_rejected(settings: AuthSettings) -> None:
    other = AuthSettings("x" * 64, "HS256", 60, settings.accounts)

    with pytest.raises(AuthError):
        decode_token(settings, issue_token(other, "ani", Role.ANALYST))


def test_tampered_token_is_rejected(settings: AuthSettings) -> None:
    token = issue_token(settings, "ani", Role.ANALYST)
    header, payload, signature = token.split(".")
    flipped = ("A" if payload[0] != "A" else "B") + payload[1:]

    with pytest.raises(AuthError):
        decode_token(settings, f"{header}.{flipped}.{signature}")


def test_token_with_algorithm_none_is_rejected(settings: AuthSettings) -> None:
    claims = {
        "sub": "ani",
        "role": "analyst",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    unsigned = jwt.encode(claims, "", algorithm="none")

    with pytest.raises(AuthError):
        decode_token(settings, unsigned)


def test_token_signed_with_a_different_hmac_algorithm_is_rejected(settings: AuthSettings) -> None:
    other = AuthSettings(SECRET, "HS512", 60, settings.accounts)

    with pytest.raises(AuthError):
        decode_token(settings, issue_token(other, "ani", Role.ANALYST))


@pytest.mark.parametrize("missing", ["sub", "role", "exp", "iat"])
def test_token_missing_a_required_claim_is_rejected(
    settings: AuthSettings, missing: str
) -> None:
    now = datetime.now(UTC)
    claims = {"sub": "ani", "role": "analyst", "iat": now, "exp": now + timedelta(minutes=5)}
    del claims[missing]
    token = jwt.encode(claims, SECRET, algorithm="HS256")

    with pytest.raises(AuthError):
        decode_token(settings, token)


def test_token_with_unknown_role_is_rejected(settings: AuthSettings) -> None:
    now = datetime.now(UTC)
    claims = {"sub": "ani", "role": "admin", "iat": now, "exp": now + timedelta(minutes=5)}
    token = jwt.encode(claims, SECRET, algorithm="HS256")

    with pytest.raises(AuthError):
        decode_token(settings, token)


def test_garbage_token_is_rejected(settings: AuthSettings) -> None:
    with pytest.raises(AuthError):
        decode_token(settings, "bukan.token.sama-sekali")


def _environment(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "JWT_SECRET_KEY": SECRET,
        "JWT_ALGORITHM": "HS256",
        "JWT_EXPIRY_MINUTES": "60",
        "DASHBOARD_ANALYST_USERNAME": "ani",
        "DASHBOARD_ANALYST_PASSWORD_HASH": "hash-analis",
        "DASHBOARD_ML_ENGINEER_USERNAME": "eko",
        "DASHBOARD_ML_ENGINEER_PASSWORD_HASH": "hash-engineer",
    } | overrides
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_load_settings_reads_both_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch)

    loaded = load_settings()

    assert loaded.accounts == {
        "ani": ("hash-analis", Role.ANALYST),
        "eko": ("hash-engineer", Role.ML_ENGINEER),
    }
    assert loaded.expiry_minutes == 60


@pytest.mark.parametrize(
    "overrides",
    [
        {"JWT_SECRET_KEY": "pendek"},
        {"JWT_SECRET_KEY": "k" * 40, "JWT_ALGORITHM": "HS512"},
        {"JWT_SECRET_KEY": "k" * 40, "JWT_ALGORITHM": "HS384"},
        {"JWT_SECRET_KEY": ""},
        {"JWT_ALGORITHM": "none"},
        {"JWT_ALGORITHM": "RS256"},
        {"JWT_EXPIRY_MINUTES": "abc"},
        {"JWT_EXPIRY_MINUTES": "0"},
        {"DASHBOARD_ANALYST_PASSWORD_HASH": ""},
        {"DASHBOARD_ML_ENGINEER_USERNAME": "ani"},
    ],
)
def test_load_settings_rejects_unsafe_or_incomplete_configuration(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]
) -> None:
    _environment(monkeypatch, **overrides)

    with pytest.raises(AuthConfigError):
        load_settings()


def test_unknown_user_check_uses_a_valid_hash() -> None:
    assert auth._unknown_user_hash().startswith("scrypt$")
