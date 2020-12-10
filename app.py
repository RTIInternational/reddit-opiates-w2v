import streamlit as st
import gensim
from pathlib import Path

models = {
    "3.8": "m.w2v",
    "3.7": "m-3.7.w2v",
    "Lowercase Unigram": "models/lowercase-sg.w2v.kv",
}

assert all((Path(p).exists() for p in models.values())), "Model Files Missing"


@st.cache(hash_funcs={gensim.models.keyedvectors.Word2VecKeyedVectors: lambda _: None})
def load_model(name):
    path = models[name]
    model = gensim.models.KeyedVectors.load(path, mmap="r")
    model.vectors_norm = model.vectors  # prevent recalc of normed vectors
    model.most_similar("opiates")  # any word will do: just to page all in
    return model


st.markdown("# Reddit Opioid Terminology - Word Embedding Explorer")

model_select = st.selectbox("Model", options=list(models.keys()))
selected_model = load_model(model_select)

terms = st.text_input("Terms (comma separated)", "oxycontin")
input_terms = [term.strip().lower() for term in terms.split(",")]
if model_select == "Lowercase scispaCy Bigrams":
    input_terms = [term.replace(" ", "__") for term in input_terms]

candidate_terms = selected_model.most_similar(input_terms, topn=10)

st.write(candidate_terms)
