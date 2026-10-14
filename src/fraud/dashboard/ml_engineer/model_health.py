"""Halaman ML engineer: kesehatan model."""

import streamlit as st

from fraud.dashboard.auth import Claims, Role, require_role


def render(claims: Claims) -> None:
    """Menampilkan kesehatan model; hanya untuk peran ML engineer."""
    require_role(claims, Role.ML_ENGINEER)
    st.header("Kesehatan model")
    st.info("Halaman ini sedang disiapkan.")
