import streamlit as st
from sentence_transformers import SentenceTransformer

# Assuming check_password.py is fixed as previously described
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


#---------------------#
# Streamlit Styling   #
#---------------------#

# Set the gradient background color
gradient_bg = "linear-gradient(135deg, #FFFFFF, #D2B48C)"  # Gradient from white to light brown (tan)

# Apply gradient background and sidebar styling using CSS
st.markdown(
    f"""
    <style>
        body {{
            background: {gradient_bg};
            color: #000000;  /* Black text color for better contrast */
        }}
        .stApp {{
            background: {gradient_bg};
            color: #000000;  /* Black text color for better contrast */
        }}
        .sidebar .sidebar-content {{
            background-color: #D2B48C;  /* Light brown color for the sidebar */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add a logo to the sidebar
logo_path = "logo.jpg"  
st.sidebar.image(logo_path, width=300, use_column_width=False)

#---------------------#
# Authentication      #
#---------------------#

if "login" not in st.session_state:
    st.session_state["login"] = False

if not check_password():
    st.stop()

#---------------------#
# Load embeddings     #
#---------------------#

# Load model
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

st.session_state['en_emb'] = load_model("all-MiniLM-L12-v2")
# st.session_state['multi_emb'] = load_model("stsb-xlm-r-multilingual")

#---------------------#
#    Main Page        #
#---------------------#

if st.session_state["login"] == False:
    st.session_state["login"] = True

st.title("ACC Chat Assistant")
st.markdown("Explore ACC data using privacy-conserving AI models from [Prediction Guard](https://www.predictionguard.com/)")
