from pathlib import Path

import gensim
import pandas as pd
import streamlit as st

models = {
    "Lowercase Unigram": "models/lowercase-sg.w2v.kv",
    "Lowercase scispaCy Unigram": "models/lowercase-sci-sg.w2v.kv",
    "Lowercase scispaCy Bigrams": "models/lowercase-sci-bigrams-sg.w2v.kv",
}

assert all((Path(p).exists() for p in models.values())), "Model Files Missing"

readme = Path("readme.md").read_text()
sl_app_text_start = readme.find("### Streamlit App")
sl_app_text_finish = readme.find("### Data")
readme = readme[:sl_app_text_start] + readme[sl_app_text_finish:]


@st.cache(hash_funcs={gensim.models.keyedvectors.Word2VecKeyedVectors: lambda _: None})
def load_model(name):
    path = models[name]
    model = gensim.models.KeyedVectors.load(path, mmap="r")
    model.most_similar("opioid")  # any word will do: just to page all in
    return model


st.markdown("# Reddit Opioid Terminology - Word Embedding Explorer")
st.markdown(
    "To use, select a model, type in a set of terms "
    "(optionally separated by commas), and select the"
    " top n number of items to return. For details on"
    " model specifics, expand the accordion below."
)

with st.beta_expander("Expand for Model Details"):
    st.markdown(readme)

st.markdown("---")
st.markdown("**Model Selection + Inputs**")
model_select = st.selectbox("Model", options=list(models.keys()))
selected_model = load_model(model_select)

terms = st.text_input("Terms (comma separated)", "oxycontin")
input_terms = [term.strip().lower() for term in terms.split(",")]

if model_select == "Lowercase scispaCy Bigrams":
    input_terms = [term.replace(" ", "__") for term in input_terms]

vocab_terms = [t for t in input_terms if t in selected_model.vocab]
oov_terms = [t for t in input_terms if t not in selected_model.vocab]


topn = st.select_slider("Return Top N=", options=[25, 50, 100, 200])
candidate_terms = selected_model.most_similar(vocab_terms, topn=topn)

if oov_terms:
    oov_str = ", ".join(oov_terms)
    st.warning(f"Input Terms Not in Model Vocabulary: {oov_str}")

st.markdown("---")

if model_select == "Lowercase scispCy Bigrams":
    candidate_terms = [(term.replace("__", " "), sim) for term, sim in candidate_terms]

in_vocab_str = ", ".join(vocab_terms)
st.markdown(
    f"Top _{topn}_ most similar terms for **{in_vocab_str}** with _{model_select}_."
)
term_df = pd.DataFrame(candidate_terms, columns=["Term", "Similarity"])
if model_select == "Lowercase scispCy Bigrams":
    term_df["Term"] = term_df["Term"].str.replace("__", " ")
st.table(term_df)
