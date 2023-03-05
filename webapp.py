import streamlit as st

from gui.app import get_app


def main():
    """Función principal de la aplicación web"""
    st.set_page_config(
        page_title="Mimic",
        page_icon="♟",
        layout="wide",
        initial_sidebar_state="auto"
    )

    if "app" not in st.session_state:
        st.session_state.app = get_app()

    st.session_state.app.render()


if __name__ == "__main__":
    main()
