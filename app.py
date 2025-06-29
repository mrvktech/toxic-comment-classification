import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
MODEL_NAMES = {
    'Logistic Regression': 'models/logisticregression_model.pkl',
    'Naive Bayes': 'models/naivebayes_model.pkl',
    'SVM': 'models/svm_model.pkl'
}

VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️")
st.title("🛡️ Toxic Comment Classifier")

model_choice = st.selectbox("Choose a model:", list(MODEL_NAMES.keys()))
model = load_model(MODEL_NAMES[model_choice])

comment = st.text_area("Enter a comment to classify:")

if st.button("Classify"):
    if comment.strip() == "":
        st.warning("⚠️ Please enter a valid comment.")
    else:
        X_vec = vectorizer.transform([comment])
        prediction = model.predict(X_vec)

        st.subheader("Prediction Results:")
        for i, label in enumerate(LABELS):
            is_toxic = prediction[0][i]
            if is_toxic:
                st.markdown(f"<span style='color: red; font-weight: bold;'>{label.capitalize()}: ✅ TOXIC</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: green; font-weight: bold;'>{label.capitalize()}: ❌ NON-TOXIC</span>", unsafe_allow_html=True)

        # # If probabilities available
        # if hasattr(model, "predict_proba"):
        #     st.subheader("Prediction Probabilities:")

        #     prob_list = model.predict_proba(X_vec)

        #     toxic_probs = []
        #     non_toxic_probs = []

        #     for prob in prob_list:
        #         # prob shape: (1, 2) → [ [non-toxic, toxic] ]
        #         non_toxic_probs.append(prob[0, 0])
        #         toxic_probs.append(prob[0, 1])

        #     prob_df = pd.DataFrame({
        #         "Label": LABELS,
        #         "Non-toxic": non_toxic_probs,
        #         "Toxic": toxic_probs
        #     })

        #     st.dataframe(prob_df.style.format("{:.4f}"))
        #     st.bar_chart(prob_df.set_index("Label")["Toxic"])
