import streamlit as st

from rag_pipeline import generate_answer

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    layout="wide"
)

st.title(
    "Hybrid Multi-Source RAG Assistant"
)

question = st.text_input(
    "Ask any question"
)

if st.button("Generate Answer"):

    result = generate_answer(
        question
    )

    st.subheader(
        "Answer"
    )

    st.write(
        result["answer"]
    )

    st.subheader(
        "Sources"
    )

    for source in result["sources"]:

        st.write(
            f"- {source}"
        )