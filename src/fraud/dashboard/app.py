"""Aplikasi dashboard: satu pintu masuk dengan halaman yang ditampilkan menurut peran login.

Dijalankan lewat: uv run streamlit run src/fraud/dashboard/app.py

Peran dibaca dari token yang lolos pemeriksaan tanda tangan dan masa berlaku di setiap
tampilan ulang, bukan dari pilihan di layar. Tiap halaman memeriksa perannya sendiri juga,
sehingga kesalahan penentuan halaman di sini tidak membuka halaman milik peran lain. Token
hanya disimpan di sesi browser yang sedang terbuka: memuat ulang halaman berarti login lagi.
"""

from collections.abc import Callable

import streamlit as st
from dotenv import load_dotenv

from fraud.dashboard.analyst import review_queue
from fraud.dashboard.auth import (
    INVALID_CREDENTIALS,
    AuthConfigError,
    AuthError,
    AuthSettings,
    Claims,
    Role,
    authenticate,
    decode_token,
    issue_token,
    load_settings,
)
from fraud.dashboard.ml_engineer import model_health

TOKEN_KEY = "token"

Page = Callable[[Claims], None]

# Daftar halaman tiap peran tidak beririsan: itulah akses eksklusif dua arah.
PAGES_BY_ROLE: dict[Role, dict[str, Page]] = {
    Role.ANALYST: {"Antrean review": review_queue.render},
    Role.ML_ENGINEER: {"Kesehatan model": model_health.render},
}

ROLE_LABELS = {Role.ANALYST: "Analis fraud", Role.ML_ENGINEER: "ML engineer"}


def current_claims(settings: AuthSettings) -> Claims | None:
    """Klaim dari token di sesi, atau None (token dibuang) kalau tidak ada atau tidak sah."""
    token = st.session_state.get(TOKEN_KEY)
    if token is None:
        return None
    try:
        return decode_token(settings, token)
    except AuthError:
        del st.session_state[TOKEN_KEY]
        return None


def show_login(settings: AuthSettings) -> None:
    st.title("Masuk")
    with st.form("login"):
        username = st.text_input("Nama pengguna")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")
    if not submitted:
        return
    role = authenticate(settings, username, password)
    if role is None:
        st.error(INVALID_CREDENTIALS)
        return
    st.session_state[TOKEN_KEY] = issue_token(settings, username, role)
    st.rerun()


def show_pages(claims: Claims) -> None:
    pages = PAGES_BY_ROLE[claims.role]
    st.sidebar.write(f"{claims.username} ({ROLE_LABELS[claims.role]})")
    if st.sidebar.button("Keluar"):
        del st.session_state[TOKEN_KEY]
        st.rerun()
    title = st.sidebar.radio("Halaman", list(pages)) if len(pages) > 1 else next(iter(pages))
    pages[title](claims)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Fraud Detection Dashboard")
    try:
        settings = load_settings()
    except AuthConfigError as error:
        st.error(f"Konfigurasi autentikasi belum benar: {error}")
        st.stop()
    claims = current_claims(settings)
    if claims is None:
        show_login(settings)
    else:
        show_pages(claims)


if __name__ == "__main__":
    main()
