# import packages
import select
import streamlit as st
import pandas as pd
import numpy as np

# page setup
st.set_page_config (
    page_title="Sentiment Analysis",
    page_icon="🔍",
    layout="centered"
)


# Global environment
#import dataset
dataset= "C:/Users/USER/PycharmProjects/TextAnalytics/Labelled_stories.txt"
with open (dataset,"r", encoding="UTF-8") as file:
    lines=file.readlines()

# clear representation of data








# Local environment

def page1():
    st.subheader("Corpora Viewer")
    if st.checkbox("Raw Data: This displays student life story in raw form"):
        st.write(lines)



def page2():
    st.subheader("Data preprocessing")


def page3():
    st.subheader("Sentiment analysis")


def page4():
    st.subheader("Evaluation")


# sidebar navigation
pages={
    "Corpora Viewer": page1,
    "data preprocessing": page2,
    "Sentiment analysis": page3,
    "Evaluation": page4
}

select_page=st.sidebar.selectbox("Select page", list(pages.keys()))
pages[select_page]()




