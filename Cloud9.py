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
    if st.checkbox("Processed Data: This displays student life story in processed form"):
        processed_lines = [line.lower() for line in lines]  # Example processing: convert to lowercase
        st.write(processed_lines)  # display the processed data
    if st.checkbox("Data Summary: This displays the summary of the data"):
        st.write(f"Total number of stories: {len(lines)}")
        # Additional summary statistics can be added here
    if st.checkbox("Data Visualization: This displays the visualization of the data"):
        st.write("Data visualization will be added here.")  # Placeholder for data visualization
    if st.checkbox("Tabular View: This displays the data in tabular form"):
        st.write("Tabular view will be added here.")  # Placeholder for tabular view
    if st.checkbox("Click to upload file"):
        st.file_uploader("Upload a file", type=["txt", "csv"])  # Allow users to upload a file


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




