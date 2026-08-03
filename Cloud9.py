# import packages
import os
import streamlit as st

# page setup
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🔍",
    layout="centered",
)

# Global environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Labelled_stories.txt")

with open(DATASET_PATH, "r", encoding="UTF-8") as file:
    lines = [line.strip() for line in file.readlines() if line.strip()]








# Local environment

def page1():
    st.subheader("Corpora Viewer")
    if st.checkbox("Raw Data: This displays student life story in raw form"):
        st.write(lines)


def page2():
    st.subheader("Data preprocessing")
    st.write("Text preprocessing steps will be added here.")


def page3():
    st.subheader("Sentiment analysis")
    st.write("Sentiment analysis results will be added here.")


def page4():
    st.subheader("Evaluation")
    st.write("Evaluation metrics will be added here.")


# sidebar navigation
pages = {
    "Corpora Viewer": page1,
    "Data preprocessing": page2,
    "Sentiment analysis": page3,
    "Evaluation": page4,
}

select_page = st.sidebar.selectbox("Select page", list(pages.keys()))
pages[select_page]()




