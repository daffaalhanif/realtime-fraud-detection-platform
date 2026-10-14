"""Halaman analis fraud: antrean review."""

import streamlit as st

from fraud.dashboard.auth import Claims, Role, require_role


def render(claims: Claims) -> None:
    """Menampilkan antrean review; hanya untuk peran analis."""
    require_role(claims, Role.ANALYST)
    st.header("Antrean review")
    st.info("Halaman ini sedang disiapkan.")
