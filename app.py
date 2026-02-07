import streamlit as st
from transformers import pipeline

st.title("🧠 Offline Generative AI Text Generator")

st.write("This project uses a locally loaded transformer model (GPT-2).")

@st.cache_resource
def load_model():
    generator = pipeline("text-generation", model="gpt2")
    return generator

generator = load_model()

user_input = st.text_area("Enter your prompt here:")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating..."):
            output = generator(
                user_input,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7
            )

            st.success("Generated Output:")
            st.write(output[0]["generated_text"])
    else:
        st.warning("Please enter some text.")
