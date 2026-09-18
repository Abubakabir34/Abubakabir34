# ============================================================
# STUDENT FEEDBACK SENTIMENT ANALYTICS
# ============================================================
import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from gensim.models import Word2Vec
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Student Feedback Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "preprocessed_text",
    "sentiment_label",
    "emotion_tag",
    "subject_specific_context"
]

DEFAULT_DATASET = (
    r"C:/Users/USER/PycharmProjects/"
    r"Abubakabir34/student_feedback_dataset.csv"
)


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


stop_words = set(stopwords.words("english"))


def preprocess_dataframe(dataframe):
    data = dataframe.copy()

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        return None, missing

    data = data[REQUIRED_COLUMNS].copy()

    # Fill missing values
    for column in REQUIRED_COLUMNS:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    # Remove empty text and labels
    data["preprocessed_text"] = (
        data["preprocessed_text"]
        .str.strip()
    )

    data["sentiment_label"] = (
        data["sentiment_label"]
        .str.strip()
    )

    data = data[
        (data["preprocessed_text"] != "") &
        (data["sentiment_label"] != "")
    ].copy()

    data = data[
        ~data["sentiment_label"]
        .str.lower()
        .isin(["unknown", "none", "null", "nan"])
    ].copy()

    # Clean text
    data["cleaned_text"] = (
        data["preprocessed_text"]
        .apply(clean_text)
    )

    # Tokenisation
    data["Tokens"] = (
        data["cleaned_text"]
        .apply(word_tokenize)
    )

    # Stopword removal
    data["Filtered_Text"] = data["Tokens"].apply(
        lambda tokens: [
            word
            for word in tokens
            if word not in stop_words
        ]
    )

    # Final preprocessed text
    data["preprocessed_text"] = (
        data["Filtered_Text"]
        .apply(lambda tokens: " ".join(tokens))
    )

    data = data[
        data["preprocessed_text"].str.strip() != ""
    ].copy()

    return data, []


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(uploaded_file):

    if uploaded_file is not None:

        try:
            return pd.read_csv(uploaded_file)

        except Exception as error:
            st.error(
                f"Could not read uploaded CSV: {error}"
            )
            st.stop()

    try:
        return pd.read_csv(DEFAULT_DATASET)

    except Exception:
        st.warning(
            "No dataset is available. "
            "Please upload a CSV file from the sidebar."
        )
        st.info(
            "Required columns: "
            "preprocessed_text, sentiment_label, "
            "emotion_tag, subject_specific_context"
        )
        st.stop()


# ============================================================
# MODEL HELPERS
# ============================================================

def get_w2v_vector(tokens, model, size=100):

    vector = np.zeros(size)
    count = 0

    for word in tokens:

        if word in model.wv:
            vector += model.wv[word]
            count += 1

    if count > 0:
        return vector / count

    return vector


def evaluate_model(
    model_name,
    model,
    X_test,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    roc_auc = np.nan

    try:

        probabilities = model.predict_proba(
            X_test
        )

        classes = model.classes_

        if len(classes) == 2:

            y_binary = (
                np.asarray(y_true) == classes[1]
            ).astype(int)

            roc_auc = roc_auc_score(
                y_binary,
                probabilities[:, 1]
            )

        else:

            roc_auc = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="weighted",
                labels=classes
            )

    except Exception:
        pass

    st.markdown(
        f"### {model_name}"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    col5.metric(
        "ROC-AUC",
        (
            f"{roc_auc:.2%}"
            if not np.isnan(roc_auc)
            else "N/A"
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# OPENROUTER LLM CLASSIFIER (zero-shot, sklearn-style wrapper)
# ============================================================

OPENROUTER_MODELS = [
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
]


class OpenRouterLLMClassifier:
    """
    Minimal scikit-learn-style wrapper around an OpenRouter chat
    model, so it can be dropped into the existing evaluate_model()
    / ROC-curve code with (almost) no changes elsewhere.

    Zero-shot: fit() just records the label set — nothing is trained.
    predict() / predict_proba() classify each text via one OpenRouter
    call and cache the result, so calling both back-to-back (as
    evaluate_model does) only costs one API call per row, not two.
    """

    def __init__(self, model_name, progress_label=None):
        self.model_name = model_name
        self.progress_label = progress_label or model_name
        self.classes_ = None
        self._client = None
        self._cache_key = None
        self._cache_preds = None
        self._cache_confidences = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Add it to a "
                    ".env file next to this script."
                )
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._client

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.Series(y).astype(str).unique()))
        return self

    def _classify_one(self, text):
        client = self._get_client()
        labels = list(self.classes_)

        system_prompt = (
            "You are a strict text classifier for student course "
            f"feedback. Classify the feedback into exactly one of "
            f"these labels: {labels}. Respond with ONLY a JSON "
            'object like {"label": "<one of the allowed labels>", '
            '"confidence": <0-1 float>}. No other text, no markdown.'
        )

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(text)[:2000]},
                ],
            )
            content = response.choices[0].message.content.strip()
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

            parsed = json.loads(content)
            label = str(parsed.get("label", labels[0]))
            confidence = float(parsed.get("confidence", 1.0 / len(labels)))

            if label not in labels:
                label = labels[0]
            confidence = min(max(confidence, 0.0), 1.0)
            return label, confidence

        except Exception:
            # Don't let one bad response (rate limit, malformed
            # JSON, network hiccup) kill the whole evaluation run.
            return labels[0], 1.0 / len(labels)

    def _run(self, X):
        texts = list(X)
        key = tuple(texts)

        if key == self._cache_key:
            return self._cache_preds, self._cache_confidences

        n = len(texts)
        preds = np.empty(n, dtype=object)
        confidences = np.empty(n, dtype=float)

        progress = st.progress(
            0.0, text=f"Classifying with {self.progress_label}..."
        )
        for i, text in enumerate(texts):
            label, confidence = self._classify_one(text)
            preds[i] = label
            confidences[i] = confidence
            progress.progress((i + 1) / n)
        progress.empty()

        self._cache_key = key
        self._cache_preds = preds
        self._cache_confidences = confidences
        return preds, confidences

    def predict(self, X):
        preds, _ = self._run(X)
        return preds

    def predict_proba(self, X):
        """
        LLMs don't give calibrated class probabilities. We ask the
        model for a self-reported confidence in its chosen label and
        spread the remaining probability mass evenly across the
        other classes — an approximation, good enough to plug into
        the existing ROC-AUC code, but not a substitute for a real
        calibrated classifier.
        """
        preds, confidences = self._run(X)
        n_classes = len(self.classes_)
        class_index = {label: i for i, label in enumerate(self.classes_)}

        proba = np.zeros((len(preds), n_classes))
        for row, (label, confidence) in enumerate(zip(preds, confidences)):
            idx = class_index[label]
            remainder = (1.0 - confidence) / max(n_classes - 1, 1)
            proba[row, :] = remainder
            proba[row, idx] = confidence

        return proba


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

def page_dashboard(df):

    st.header("📊 Student Feedback Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Feedback",
        f"{len(df):,}"
    )

    col2.metric(
        "Sentiment Classes",
        str(df["sentiment_label"].nunique())
    )

    col3.metric(
        "Emotion Categories",
        str(df["emotion_tag"].nunique())
    )

    col4.metric(
        "Subjects / Contexts",
        str(
            df["subject_specific_context"]
            .nunique()
        )
    )

    st.markdown("---")

    st.subheader("📈 Sentiment Distribution")

    distribution = (
        df["sentiment_label"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        x=distribution.index,
        y=distribution.values,
        ax=ax
    )

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Feedback Records")
    ax.set_title("Student Sentiment Distribution")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df[
            REQUIRED_COLUMNS
        ].head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 2 — DATASET PROCESSING
# ============================================================

def page_dataset(df):

    st.header("📁 Dataset Processing")

    st.subheader("Required CSV Columns")

    st.code(
        "preprocessed_text\n"
        "sentiment_label\n"
        "emotion_tag\n"
        "subject_specific_context",
        language="text"
    )

    if uploaded_file is not None:
        st.success(
            f"Currently processing: {uploaded_file.name}"
        )

    st.subheader("Processed Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("📥 Download")

    processed_csv = (
        df[REQUIRED_COLUMNS]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Processed CSV",
        data=processed_csv,
        file_name="processed_student_feedback.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("🔎 Text Preprocessing Example")

    preview_columns = [
        "preprocessed_text",
        "sentiment_label"
    ]

    st.dataframe(
        df[preview_columns].head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 3 — TEXT PREPROCESSING
# ============================================================

def page_preprocessing(df):

    st.header("🧹 Text Preprocessing")

    st.markdown(
        "The uploaded feedback is cleaned, tokenised, "
        "and stopwords are removed before model training."
    )

    st.subheader("1. Original Dataset")

    st.dataframe(
        df[REQUIRED_COLUMNS].head(20),
        use_container_width=True
    )

    st.subheader("2. Cleaned Text")

    st.dataframe(
        df[
            [
                "preprocessed_text",
                "cleaned_text",
                "sentiment_label"
            ]
        ].head(20),
        use_container_width=True
    )

    st.subheader("3. Tokenisation")

    st.dataframe(
        df[
            [
                "cleaned_text",
                "Tokens",
                "sentiment_label"
            ]
        ].head(20),
        use_container_width=True
    )

    st.subheader("4. Stopword Removal")

    st.dataframe(
        df[
            [
                "Tokens",
                "Filtered_Text",
                "sentiment_label"
            ]
        ].head(20),
        use_container_width=True
    )

    st.subheader("5. Final Preprocessed Text")

    st.dataframe(
        df[
            [
                "preprocessed_text",
                "sentiment_label"
            ]
        ].head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

def page_model_performance(df):

    st.header("🤖 Model Performance")

    if df["sentiment_label"].nunique() < 2:
        st.error(
            "The dataset must contain at least "
            "two sentiment classes."
        )
        return

    # Avoid a train/test split that cannot represent classes.
    try:

        X_train_text, X_test_text, y_train, y_test = (
            train_test_split(
                df["preprocessed_text"],
                df["sentiment_label"],
                test_size=0.20,
                random_state=42,
                stratify=df["sentiment_label"]
            )
        )

    except ValueError:

        X_train_text, X_test_text, y_train, y_test = (
            train_test_split(
                df["preprocessed_text"],
                df["sentiment_label"],
                test_size=0.20,
                random_state=42
            )
        )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    st.subheader("TF-IDF Feature Extraction")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(
        X_train_text
    )

    X_test_tfidf = tfidf.transform(
        X_test_text
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = SVC(
        kernel="linear",
        probability=True,
        random_state=42
    )

    svm_model.fit(
        X_train_tfidf,
        y_train
    )

    y_pred_svm = svm_model.predict(
        X_test_tfidf
    )

    svm_results = evaluate_model(
        "SVM - TF-IDF",
        svm_model,
        X_test_tfidf,
        y_test,
        y_pred_svm
    )

    # --------------------------------------------------------
    # WORD2VEC
    # --------------------------------------------------------

    st.subheader("Word2Vec Feature Extraction")

    token_sentences = (
        df["Filtered_Text"]
        .tolist()
    )

    w2v_model = Word2Vec(
        sentences=token_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )

    X_w2v = np.array([
        get_w2v_vector(
            tokens,
            w2v_model,
            100
        )
        for tokens in df["Filtered_Text"]
    ])

    train_indices = X_train_text.index
    test_indices = X_test_text.index

    X_train_w2v = X_w2v[
        df.index.get_indexer(train_indices)
    ]

    X_test_w2v = X_w2v[
        df.index.get_indexer(test_indices)
    ]

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_w2v,
        y_train
    )

    y_pred_rf = rf_model.predict(
        X_test_w2v
    )

    rf_results = evaluate_model(
        "Random Forest - Word2Vec",
        rf_model,
        X_test_w2v,
        y_test,
        y_pred_rf
    )

    # --------------------------------------------------------
    # LLM (OPENROUTER) — opt-in, since each row is a paid API call
    # --------------------------------------------------------

    st.subheader("🧠 LLM Zero-Shot Classification (OpenRouter)")

    llm_enabled = st.checkbox(
        "Include an LLM-based classifier in the comparison",
        value=False,
        help=(
            "Calls an OpenRouter model once per test example. "
            "Off by default to avoid unexpected API cost."
        )
    )

    llm_results = None
    llm_model = None
    X_test_llm = None

    if llm_enabled:

        col_a, col_b = st.columns(2)

        with col_a:
            llm_model_name = st.selectbox(
                "OpenRouter model",
                OPENROUTER_MODELS,
                index=0
            )

        with col_b:
            max_possible = len(X_test_text)
            default_sample = min(40, max_possible)
            llm_sample_size = st.slider(
                "Test rows to classify",
                min_value=min(10, max_possible),
                max_value=max_possible,
                value=default_sample,
                step=5,
                help=(
                    "LLM evaluation runs on a random sample of the "
                    "test set (not all of it) to keep cost and "
                    "runtime reasonable."
                )
            )

        cache_key = (
            f"llm_eval::{llm_model_name}::{llm_sample_size}"
            f"::{len(df)}"
        )

        run_llm = st.button(
            "🚀 Run LLM Evaluation",
            use_container_width=True
        )

        if run_llm:

            if not os.environ.get("OPENROUTER_API_KEY"):
                st.error(
                    "OPENROUTER_API_KEY is not set. Add it to a "
                    ".env file next to this script and restart it."
                )

            else:
                sample_index = (
                    X_test_text
                    .sample(n=llm_sample_size, random_state=42)
                    .index
                )

                X_sample = X_test_text.loc[sample_index]
                y_sample = y_test.loc[sample_index]

                llm_model = OpenRouterLLMClassifier(
                    llm_model_name,
                    progress_label=llm_model_name
                )
                llm_model.fit(X_train_text, y_train)

                y_pred_llm = llm_model.predict(X_sample)

                st.session_state[cache_key] = {
                    "model_name": llm_model_name,
                    "sample_size": llm_sample_size,
                    "model": llm_model,
                    "X_sample": X_sample,
                    "y_sample": y_sample,
                    "y_pred": y_pred_llm,
                }

        cached = st.session_state.get(cache_key)

        if cached is not None:

            st.caption(
                f"Results below are based on {cached['sample_size']} "
                f"randomly sampled test rows (out of {max_possible} "
                f"total), classified by **{cached['model_name']}** "
                "via OpenRouter — not the full test set used for "
                "SVM and Random Forest above, so treat this as "
                "directional rather than a strict apples-to-apples "
                "comparison."
            )

            llm_results = evaluate_model(
                f"LLM ({cached['model_name']})",
                cached["model"],
                cached["X_sample"],
                cached["y_sample"],
                cached["y_pred"]
            )

            llm_model = cached["model"]
            X_test_llm = cached["X_sample"]

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    st.subheader("📊 Model Comparison")

    all_results = [svm_results, rf_results]
    if llm_results is not None:
        all_results.append(llm_results)

    comparison = pd.DataFrame(all_results)

    display_comparison = comparison.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        display_comparison[column] = (
            display_comparison[column] * 100
        ).round(2)

    display_comparison = (
        display_comparison
        .rename(
            columns={
                "Accuracy": "Accuracy (%)",
                "Precision": "Precision (%)",
                "Recall": "Recall (%)",
                "F1 Score": "F1 Score (%)",
                "ROC-AUC": "ROC-AUC (%)"
            }
        )
    )

    st.dataframe(
        display_comparison,
        use_container_width=True
    )

    best_model = comparison.loc[
        comparison["F1 Score"].idxmax(),
        "Model"
    ]

    st.success(
        f"🏆 Best model based on weighted F1: "
        f"**{best_model}**"
    )

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    st.subheader("📈 ROC-AUC Curves")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    roc_models = [
        (
            "SVM - TF-IDF",
            svm_model,
            X_test_tfidf,
            y_test
        ),
        (
            "Random Forest - Word2Vec",
            rf_model,
            X_test_w2v,
            y_test
        )
    ]

    if llm_results is not None and llm_model is not None:
        # The LLM was evaluated on a sampled subset, so it needs its
        # own ground-truth labels — not the full-test-set y_test.
        roc_models.append(
            (
                f"LLM ({llm_model.model_name})",
                llm_model,
                X_test_llm,
                cached["y_sample"]
            )
        )

    for model_name, model, X_features, y_true_for_model in roc_models:

        try:

            probabilities = (
                model.predict_proba(
                    X_features
                )
            )

            classes = model.classes_

            if len(classes) == 2:

                y_binary = (
                    np.asarray(y_true_for_model)
                    == classes[1]
                ).astype(int)

                if len(np.unique(y_binary)) >= 2:

                    fpr, tpr, _ = roc_curve(
                        y_binary,
                        probabilities[:, 1]
                    )

                    curve_auc = auc(
                        fpr,
                        tpr
                    )

                    ax.plot(
                        fpr,
                        tpr,
                        label=(
                            f"{model_name} "
                            f"(AUC = {curve_auc:.3f})"
                        )
                    )

            else:

                y_test_array = np.asarray(
                    y_true_for_model
                )

                for i, class_label in enumerate(
                    classes
                ):

                    y_class = (
                        y_test_array
                        == class_label
                    ).astype(int)

                    if len(
                        np.unique(y_class)
                    ) < 2:
                        continue

                    fpr, tpr, _ = roc_curve(
                        y_class,
                        probabilities[:, i]
                    )

                    curve_auc = auc(
                        fpr,
                        tpr
                    )

                    ax.plot(
                        fpr,
                        tpr,
                        label=(
                            f"{model_name} - "
                            f"{class_label} "
                            f"(AUC = "
                            f"{curve_auc:.3f})"
                        )
                    )

        except Exception as error:

            st.warning(
                f"Could not plot ROC for "
                f"{model_name}: {error}"
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC-AUC Curves"
    )

    ax.legend(
        loc="lower right"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PAGE 5 — SENTIMENT PREDICTION
# ============================================================

def page_prediction(df):

    st.header("🔮 Student Sentiment Prediction")

    user_text = st.text_area(
        "Enter student feedback:",
        height=160
    )

    uploaded_prediction_file = st.file_uploader(
        "Or upload TXT/CSV for prediction",
        type=["txt", "csv"],
        key="prediction_upload"
    )

    file_content = ""

    if uploaded_prediction_file is not None:

        try:

            if uploaded_prediction_file.name.lower().endswith(
                ".csv"
            ):

                prediction_df = pd.read_csv(
                    uploaded_prediction_file
                )

                if (
                    "preprocessed_text"
                    in prediction_df.columns
                ):

                    file_content = " ".join(
                        prediction_df[
                            "preprocessed_text"
                        ]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                else:

                    file_content = (
                        prediction_df
                        .astype(str)
                        .to_string(index=False)
                    )

            else:

                file_content = (
                    uploaded_prediction_file
                    .read()
                    .decode("utf-8")
                )

            st.success(
                "Prediction file loaded successfully."
            )

        except Exception as error:

            st.error(
                f"Could not process prediction file: "
                f"{error}"
            )

    final_input = (
        file_content
        if file_content.strip()
        else user_text
    )

    col1, col2 = st.columns(2)

    with col1:

        predict = st.button(
            "🔍 Predict Sentiment",
            use_container_width=True
        )

    with col2:

        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear:
        st.rerun()

    if predict:

        if not final_input.strip():

            st.warning(
                "Please enter text or upload a file."
            )
            return

        if df["sentiment_label"].nunique() < 2:

            st.error(
                "At least two sentiment classes are required."
            )
            return

        try:

            # Train prediction model on all available data.
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )

            X = vectorizer.fit_transform(
                df["preprocessed_text"]
            )

            model = SVC(
                kernel="linear",
                probability=True,
                random_state=42
            )

            model.fit(
                X,
                df["sentiment_label"]
            )

            cleaned = clean_text(
                final_input
            )

            tokens = word_tokenize(
                cleaned
            )

            filtered = [
                word
                for word in tokens
                if word not in stop_words
            ]

            processed = " ".join(
                filtered
            )

            input_features = vectorizer.transform(
                [processed]
            )

            prediction = model.predict(
                input_features
            )[0]

            probabilities = (
                model.predict_proba(
                    input_features
                )[0]
            )

            confidence = float(
                np.max(probabilities)
            )

            st.success(
                f"### Predicted Sentiment: "
                f"**{prediction}**"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# SIDEBAR NAVIGATION — SAME PATTERN AS YOUR EXAMPLE
# ============================================================

with st.sidebar:

    st.title("🎓 Student Analytics")
    st.caption(
        "Student Feedback Sentiment Dashboard"
    )

    st.markdown("---")

    st.subheader("🧭 Navigation")

    pages = {
        "📊 Dashboard": page_dashboard,
        "🧹 Data Preprocessing": page_preprocessing,
        "🤖 Model Performance": page_model_performance,
        "🔮 Sentiment Prediction": page_prediction,
        "📁 Dataset Processing": page_dataset
    }

    select_page = st.radio(
        "Select a page",
        list(pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("📁 Upload CSV")

    uploaded_file = st.file_uploader(
        "Choose a student feedback CSV",
        type=["csv"],
        help=(
            "Required columns: preprocessed_text, "
            "sentiment_label, emotion_tag, "
            "subject_specific_context"
        )
    )

    if uploaded_file is not None:

        st.success(
            f"✅ {uploaded_file.name}"
        )

        st.caption(
            f"Size: "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

    else:

        st.info(
            "Upload a CSV to analyze student feedback."
        )

    st.markdown("---")

    st.subheader("⚙️ File Information")

    if uploaded_file is not None:
        st.caption(
            "The uploaded CSV is currently active."
        )
    else:
        st.caption(
            "The default local dataset will be used "
            "if available."
        )


# ============================================================
# LOAD + VALIDATE + PREPROCESS DATA
# ============================================================

raw_df = load_dataset(
    uploaded_file
)

df, missing_columns = preprocess_dataframe(
    raw_df
)

if missing_columns:

    st.error(
        "The uploaded CSV is missing required columns:"
    )

    st.write(
        missing_columns
    )

    st.write(
        "Columns found:"
    )

    st.write(
        raw_df.columns.tolist()
    )

    st.stop()

if df.empty:

    st.error(
        "No usable records remain after processing."
    )

    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🎓 Student Feedback Analytics"
)

st.caption(
    "NLP-based student feedback analysis using "
    "TF-IDF, Word2Vec, SVM and Random Forest."
)


# ============================================================
# DISPLAY SELECTED PAGE
# ============================================================

pages[select_page](df)
