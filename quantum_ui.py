import streamlit as st

def apply_quantum_theme():

    st.markdown(
    """
    <style>

    /* -------------------------
    Main App Background
    ------------------------- */

    .stApp {
        background: radial-gradient(circle at 20% 20%, #0f2027, #050b18 60%);
        background-attachment: fixed;
        color: white;
    }

    /* -------------------------
    Quantum Particle Grid
    ------------------------- */

    .stApp::before {
        content: "";
        position: fixed;
        top:0;
        left:0;
        width:100%;
        height:100%;
        background-image:
            radial-gradient(#00ffff 1px, transparent 1px);
        background-size: 40px 40px;
        opacity:0.15;
        pointer-events:none;
    }

    /* -------------------------
    Dashboard Container
    ------------------------- */

    .block-container{
        background: rgba(0,0,0,0.75);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 0 20px #00ffff33;
    }

    /* -------------------------
    Titles
    ------------------------- */

    h1, h2, h3 {
        color:#00ffff;
        text-shadow:0px 0px 8px #00ffff;
    }

    /* -------------------------
    Sidebar
    ------------------------- */

    section[data-testid="stSidebar"]{
        background: #020617;
        border-right:1px solid #00ffff33;
    }

    /* -------------------------
    Buttons
    ------------------------- */

    button[kind="primary"]{
        background:#00ffff;
        color:black;
        border-radius:10px;
        box-shadow:0 0 10px #00ffff;
    }

    /* -------------------------
    Glowing Charts
    ------------------------- */

    .js-plotly-plot {
        box-shadow:0 0 15px #00ffff55;
        border-radius:10px;
    }

    </style>
    """,
    unsafe_allow_html=True
    )


def quantum_loader():

    st.markdown(
    """
    <style>
    .loader {
      border: 6px solid #222;
      border-top: 6px solid #00ffff;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin:auto;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    </style>

    <div class="loader"></div>
    """,
    unsafe_allow_html=True
    )