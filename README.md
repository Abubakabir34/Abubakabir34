# Sentiment Analysis App

This project is a Streamlit-based sentiment analysis app that can be deployed on platforms like Render, Railway, or Heroku.

## Run locally

```bash
streamlit run app.py
```

## Deployment notes

- The app uses a repository-local dataset at [data/Labelled_stories.txt](data/Labelled_stories.txt)
- The deployment entry point is [app.py](app.py)
- The hosting command is defined in [Procfile](Procfile)
