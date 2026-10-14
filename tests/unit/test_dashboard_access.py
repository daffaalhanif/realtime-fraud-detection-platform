"""Uji akses dashboard: tiap peran hanya melihat halamannya, di kedua arah."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fraud.dashboard import app as app_module
from fraud.dashboard.analyst import review_queue
from fraud.dashboard.auth import (
    INVALID_CREDENTIALS,
    AuthSettings,
    Claims,
    ForbiddenError,
    Role,
    hash_password,
    issue_token,
)
from fraud.dashboard.ml_engineer import model_health

APP_PATH = str(Path(app_module.__file__))
SECRET = "k" * 64
PASSWORDS = {"ani": "analis-password-1", "eko": "engineer-password-1"}

ANALYST_HEADER = "Antrean review"
ENGINEER_HEADER = "Kesehatan model"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> AuthSettings:
    hashes = {name: hash_password(password) for name, password in PASSWORDS.items()}
    # Uji hanya memakai data buatan ini, bukan berkas .env yang asli.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for name, value in {
        "JWT_SECRET_KEY": SECRET,
        "JWT_ALGORITHM": "HS256",
        "JWT_EXPIRY_MINUTES": "60",
        "DASHBOARD_ANALYST_USERNAME": "ani",
        "DASHBOARD_ANALYST_PASSWORD_HASH": hashes["ani"],
        "DASHBOARD_ML_ENGINEER_USERNAME": "eko",
        "DASHBOARD_ML_ENGINEER_PASSWORD_HASH": hashes["eko"],
    }.items():
        monkeypatch.setenv(name, value)
    return AuthSettings(
        secret=SECRET,
        algorithm="HS256",
        expiry_minutes=60,
        accounts={
            "ani": (hashes["ani"], Role.ANALYST),
            "eko": (hashes["eko"], Role.ML_ENGINEER),
        },
    )


def _open() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def _login(username: str, password: str) -> AppTest:
    at = _open()
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click()
    return at.run()


def _headers(at: AppTest) -> list[str]:
    return [header.value for header in at.header]


def _shows_login(at: AppTest) -> bool:
    return len(at.text_input) == 2 and not at.header


def test_visitor_without_login_sees_only_the_login_form(settings: AuthSettings) -> None:
    at = _open()

    assert not at.exception
    assert _shows_login(at)
    assert _headers(at) == []


def test_analyst_sees_the_analyst_page_and_not_the_ml_engineer_page(
    settings: AuthSettings,
) -> None:
    at = _login("ani", PASSWORDS["ani"])

    assert not at.exception
    assert _headers(at) == [ANALYST_HEADER]
    assert ENGINEER_HEADER not in _headers(at)
    # Hanya satu halaman, jadi tidak ada pilihan halaman lain yang ditawarkan.
    assert len(at.sidebar.radio) == 0


def test_ml_engineer_sees_the_ml_engineer_page_and_not_the_analyst_page(
    settings: AuthSettings,
) -> None:
    at = _login("eko", PASSWORDS["eko"])

    assert not at.exception
    assert _headers(at) == [ENGINEER_HEADER]
    assert ANALYST_HEADER not in _headers(at)
    assert len(at.sidebar.radio) == 0


def test_wrong_password_stays_on_login_with_a_generic_message(settings: AuthSettings) -> None:
    at = _login("ani", "password-salah")

    assert _shows_login(at)
    assert [error.value for error in at.error] == [INVALID_CREDENTIALS]
    assert "token" not in at.session_state


def test_unknown_user_gets_the_same_message_as_a_wrong_password(settings: AuthSettings) -> None:
    at = _login("tidak-ada", PASSWORDS["ani"])

    assert _shows_login(at)
    assert [error.value for error in at.error] == [INVALID_CREDENTIALS]


def test_each_role_page_refuses_the_other_role_directly(settings: AuthSettings) -> None:
    analyst = Claims("ani", Role.ANALYST)
    engineer = Claims("eko", Role.ML_ENGINEER)

    with pytest.raises(ForbiddenError):
        model_health.render(analyst)
    with pytest.raises(ForbiddenError):
        review_queue.render(engineer)


def test_page_registry_has_no_page_shared_between_roles() -> None:
    analyst_pages = set(app_module.PAGES_BY_ROLE[Role.ANALYST].values())
    engineer_pages = set(app_module.PAGES_BY_ROLE[Role.ML_ENGINEER].values())

    assert analyst_pages.isdisjoint(engineer_pages)
    assert set(app_module.PAGES_BY_ROLE) == set(Role)


def test_expired_token_in_session_returns_to_login(settings: AuthSettings) -> None:
    long_ago = datetime.now(UTC) - timedelta(hours=3)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["token"] = issue_token(settings, "ani", Role.ANALYST, now=long_ago)

    at.run()

    assert _shows_login(at)
    assert "token" not in at.session_state


@pytest.mark.parametrize("token", ["bukan-token", "a.b.c", ""])
def test_garbage_token_in_session_returns_to_login(settings: AuthSettings, token: str) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["token"] = token

    at.run()

    assert _shows_login(at)
    assert "token" not in at.session_state


def test_token_signed_with_another_key_returns_to_login(settings: AuthSettings) -> None:
    forged = AuthSettings("x" * 64, "HS256", 60, settings.accounts)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["token"] = issue_token(forged, "ani", Role.ANALYST)

    at.run()

    assert _shows_login(at)


def test_logout_clears_the_session_and_returns_to_login(settings: AuthSettings) -> None:
    at = _login("eko", PASSWORDS["eko"])
    assert _headers(at) == [ENGINEER_HEADER]

    at.sidebar.button[0].click()
    at.run()

    assert _shows_login(at)
    assert "token" not in at.session_state


def test_unsafe_configuration_shows_a_message_and_no_login(
    settings: AuthSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "pendek")

    at = _open()

    assert len(at.error) == 1
    assert "JWT_SECRET_KEY" in at.error[0].value
    assert len(at.text_input) == 0
