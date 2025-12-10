import streamlit as st
import pickle

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('toxicity_model.pkl', 'rb') as f:
    model = pickle.load(f)

THRESHOLD = 10  # toxicity threshold (in %)

def polite_suggestion(comment):
    if "bad" in comment.lower():
        return "Consider rephrasing to something more positive."
    if "horrible" in comment.lower() or "stupid" in comment.lower():
        return "Try to express your opinion politely."
    return "Try rephrasing to a more polite tone."

# Streamlit UI
st.title("AI Toxic Text Filter")

comment = st.text_input("Enter your message:")

if st.button("Analyze"):
    if comment.strip() == "":
        st.warning("Please enter a message!")
    else:
        comment_vec = vectorizer.transform([comment])
        score = model.predict_proba(comment_vec)[0][1] * 100

        st.write(f"### Toxicity score: `{score:.2f}%`")

        if score >= THRESHOLD:
            st.error("Status: **Toxic**")
            st.write("### Suggested polite rewrite:")
            st.info(polite_suggestion(comment))
        else:
            st.success("Status: **Safe**")
            st.write("No change needed.")

