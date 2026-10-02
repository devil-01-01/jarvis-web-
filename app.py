import streamlit as st
from groq import Groq
import urllib.parse

st.set_page_config(page_title="JARVIS ULTIMATE", page_icon="🤖")
st.title("🤖 JARVIS ULTIMATE")

GROQ_KEY = st.secrets.get("GROQ_KEY", None) if hasattr(st, "secrets") else None
if not GROQ_KEY:
        GROQ_KEY = st.sidebar.text_input("Groq API Key", type="password")

        client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

        if "history" not in st.session_state:
                st.session_state.history = []

                for m in st.session_state.history:
                        with st.chat_message(m["role"]):
                                    st.markdown(m["content"])

                                    prompt = st.chat_input("Pucho kuch...")
                                    if prompt:
                                            st.session_state.history.append({"role": "user", "content": prompt})
                                                with st.chat_message("user"):
                                                            st.markdown(prompt)
                                                                if client:
                                                                            with st.chat_message("assistant"):
                                                                                            r = client.chat.completions.create(model="openai/gpt-oss-20b", messages=st.session_state.history)
                                                                                                        ans = r.choices[0].message.content
                                                                                                                    st.markdown(ans)
                                                                                                                                st.session_state.history.append({"role": "assistant", "content": ans})

                                                                                                                                st.divider()
                                                                                                                                st.subheader("🎨 Image Generator")
                                                                                                                                p = st.text_input("Image prompt")
                                                                                                                                if st.button("Generate Image"):
                                                                                                                                        if p:
                                                                                                                                                    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}"
                                                                                                                                                            st.image(url)