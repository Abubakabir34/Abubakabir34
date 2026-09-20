# Student Feedback Analytics

This is a Streamlit app for exploring student feedback, comparing sentiment models, and predicting sentiment for new feedback.

## Run locally

```bash
streamlit run app.py
```

The app includes a small repository-local sample dataset, so it opens immediately after installation. Use the sidebar to upload a CSV with these columns:

```text
preprocessed_text, sentiment_label, emotion_tag, subject_specific_context
```

## Deployment notes

- The app uses a repository-local dataset at [data/Labelled_stories.txt](data/Labelled_stories.txt)
- The deployment entry point is [app.py](app.py)
- The hosting command is defined in [Procfile](Procfile)
- Set `OPENROUTER_API_KEY` in the host's environment variables to enable AI-generated summaries; the core analytics work without it

### Deploy on Render

Create a new **Web Service** from this repository with:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Environment variable:** `OPENROUTER_API_KEY` (optional)

The service will provide a public URL after the first successful deploy. Keep the repository and dataset available in the deployed service because the default dashboard reads the bundled sample file.
Once deployed, the URL: you never commit it, Streamlit generates it for you.
