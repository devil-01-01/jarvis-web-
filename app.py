import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Page setup
st.set_page_config(page_title="Mini ChatGPT - Torch Transformer", page_icon="🤖")
st.title("🤖 Mini ChatGPT - Torch + Transformer")
st.write("Ye real Transformer model hai, jo Torch se bana hai. Pydroid wale se 100x bada hai.")

# Model load - ek baar hi hoga
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small" # 117M parameters, Transformer model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Transformer model load ho raha hai... pehli baar 500MB download hoga"):
    tokenizer, model = load_model()

st.success("Model ready hai!")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Purani chat dikhao
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Naya message
user_input = st.chat_input("Kuch bhi pucho...")

if user_input:
    # User ka message dikhao
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Transformer se jawab generate karo
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hu..."):
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

            if st.session_state.chat_history_ids is not None:
                bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
            else:
                bot_input_ids = new_input_ids

            # Transformer generation
            st.session_state.chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8
            )

            response = tokenizer.decode(
                st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            st.write(response)

    st.session_state.history.append({"role": "assistant", "content": response})

# Sidebar - Info
with st.sidebar:
    st.header("Model Info")
    st.write("**Model:** DialoGPT-small")
    st.write("**Type:** Transformer")
    st.write("**Library:** Torch (PyTorch)")
    st.write("**Parameters:** 117 Million")
    st.write("**Framework:** Streamlit")
    if st.button("Chat Clear Karo"):
        st.session_state.history = []
        st.session_state.chat_history_ids = None
        st.rerun()
