import streamlit as st
from groq import Groq
import sympy as sp
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="JARVIS - Your AI", page_icon="🤖", layout="wide")
st.title("🤖 JARVIS - Web Edition")
st.caption("Tumhara banaya hua AI - Ab Web pe Live")

# --- API KEY ---
# Streamlit Cloud me Secrets me GROQ_KEY daalna
try:
    GROQ_KEY = st.secrets["GROQ_KEY"]
except:
    GROQ_KEY = st.sidebar.text_input("Groq API Key dalo (gsk_...)", type="password")

client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# --- MEMORY ---
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = []

# --- SIDEBAR - BADA WALA FEATURE ---
with st.sidebar:
    st.header("JARVIS Control Panel")
    st.write(f"Time: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    if st.button("🗑️ Chat Clear Karo"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.write("**Capabilities:**")
    st.write("✅ Maths Solve")
    st.write("✅ Coding")
    st.write("✅ Chat Memory")
    st.write("✅ Web pe 24x7")

# --- MAIN BRAIN - WAHI BADA WALA LOGIC ---
def jarvis_brain(question, chat_history):
    try:
        # 1. MATHS LOGIC - SYMPY
        if "solve" in question.lower() and "x" in question.lower():
            x = sp.symbols('x')
            clean = question.lower().replace("solve","").replace("=","=").replace("^","**").strip()
            if "=" in clean:
                l, r = clean.split("=")
                eq = sp.sympify(l) - sp.sympify(r)
                sol = sp.solve(eq, x)
                return f"**Math Solved Boss:** Equation `{question}` ka solution hai: **x = {sol}**"

        # 2. AI LOGIC - GROQ
        messages = [{"role": "system", "content": "You are JARVIS, created by Boss. You are very helpful, smart, a bit funny. You speak in Hindi + English mix like user. You are Boss's personal AI assistant."}]

        # Last 6 chats yaad rakho
        for role, content in chat_history[-6:]:
            messages.append({"role": "user" if role=="Boss" else "assistant", "content": content})

        messages.append({"role": "user", "content": question})

        if not client:
            return "Boss API Key dalo sidebar me!"

        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7
        )
        return res.choices[0].message.content

    except Exception as e:
        return f"Error Boss: {e}. Check karo key sahi hai ya nahi."

# --- CHAT DISPLAY ---
for role, msg in st.session_state.history:
    with st.chat_message("user" if role=="Boss" else "assistant"):
        st.markdown(f"**{role}:** {msg}")

# --- INPUT ---
q = st.chat_input("Boss, kuch bolo...")

if q:
    with st.chat_message("user"):
        st.markdown(f"**Boss:** {q}")
    st.session_state.history.append(("Boss", q))

    ans = jarvis_brain(q, st.session_state.history)

    with st.chat_message("assistant"):
        st.markdown(f"**JARVIS:** {ans}")
    st.session_state.history.append(("JARVIS", ans))
