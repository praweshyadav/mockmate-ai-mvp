import os
import json
import time
from datetime import datetime
from pathlib import Path
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import re


HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")

if not HF_TOKEN:
    HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_KEY")
    
# ..................
# ADD THESE IMPORTS
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text
# ..................

st.set_page_config(page_title="MockMate AI", page_icon="🎤", layout="centered")

# -----------------------------
# Configuration
# -----------------------------
client = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.2,
     huggingfacehub_api_token=HF_TOKEN
)

MODEL = ChatHuggingFace(llm=client)



QUESTIONS = [
    "Tell me about yourself and the kind of role you are targeting.",
    "Describe one technical project you are proud of. What problem did it solve and what did you personally build?",
    "Tell me about a difficult technical problem you faced. How did you approach it?",
    "Why should we hire you for this role?",
    "Do you have any questions for the interviewer?"
]


SYSTEM = """
You are MockMate AI, an AI interviewer helping college students practice interviews.

IMPORTANT RULES:
1. You are an AI. Never pretend to be a human interviewer.
2. Evaluate ONLY the candidate's actual answer.
3. NEVER invent, assume, or add candidate information.
4. NEVER create projects, degrees, companies, skills, achievements, locations,
   experience, technologies, or other personal details that are not explicitly
   present in the candidate's answer.
5. If information is missing, say that it is missing instead of guessing.
6. Do not rewrite the candidate's answer by adding fictional achievements.
7. In BETTER VERSION, improve the candidate's answer using ONLY facts they provided.
   You may improve grammar, structure, clarity, and wording, but do not add new facts.

Evaluate the answer based on:
- Relevance to the question
- Clarity and structure
- Specificity
- Evidence/examples
- Communication quality
- Technical correctness when applicable

Return feedback in exactly this structure:

SCORE: <integer from 0-10>

WHAT WORKED:
- <specific point from the candidate's actual answer>
- <specific point from the candidate's actual answer>

NEEDS IMPROVEMENT:
- <specific improvement>
- <specific improvement>

BETTER VERSION:
<Rewrite the candidate's answer using ONLY information actually provided.
If important information is missing, use a placeholder such as [add your project/result here].>

NEXT TIP:
<one specific actionable tip>
"""

# ADD THIS FUNCTION (place below SYSTEM)....................................

def speak_text(text):
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <script>
            const msg = new SpeechSynthesisUtterance(`{safe_text}`);
            msg.rate = 1;
            msg.pitch = 1;
            msg.lang = "en-US";
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )



def ai_call(prompt: str) -> str:
    response = MODEL.invoke([
        ("system", SYSTEM),
        ("human", prompt)
    ])

    return response.content

def parse_score(text):
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text, re.I)
    return float(m.group(1)) if m else None

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "started": False,
    "q_index": 0,
    "answers": [],
    "feedback": [],
    "start_time": None,
    "finished": False,
    "role": "Software / AI-ML Engineer",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Header / avatar
# -----------------------------
st.markdown(
    """
    <style>
    .avatar {
        width: 105px; height: 105px; border-radius: 50%;
        margin: 0 auto 10px auto; display:flex; align-items:center;
        justify-content:center; font-size:58px;
        border: 3px solid #1f4b73;
        background: #eaf3fb;
    }
    .center { text-align:center; }
    .small { color:#667085; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="avatar">🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="center"><h1>Practice interviews with an AI interviewer</h1><p>Practice interviews with an AI interviewer — then get instant, structured feedback.</p></div>', unsafe_allow_html=True)
st.info("You are interacting with an AI, not a human interviewer. Do not enter confidential or sensitive personal information.")

if not st.session_state.started and not st.session_state.finished:
    st.subheader("Start a 5-question practice interview")
    st.session_state.role = st.selectbox(
        "Target role",
        ["Software / AI-ML Engineer", "Data Engineer", "Data Scientist", "Product / Tech Intern", "Other"],
        index=0
    )
    st.write("**How it works:** The avatar asks one question at a time. You answer, receive feedback, and continue.")
    if st.button("🚀 Start Interview", use_container_width=True):
        st.session_state.started = True
        st.session_state.start_time = time.time()
        st.rerun()

elif st.session_state.started and not st.session_state.finished:
    idx = st.session_state.q_index
    question = QUESTIONS[idx]

    st.progress((idx + 1 )/ len(QUESTIONS))
    st.caption(f"Question {idx+1} of {len(QUESTIONS)} • Target role: {st.session_state.role}")
    # st.markdown(f"### 🤖 Interviewer")
    # st.write(question)
    col1, col2 = st.columns([10,1])

    with col1:
        st.markdown("### 🤖 MockMate AI")
        st.write(question)

    with col2:
        if st.button("🔊", key=f"q_speak_{idx}"):
            speak_text(question)
            # ......................................change apply.........................................
    # 🎙️ Voice input + ⌨️ Text input
    st.write("Your answer")

    spoken_text = speech_to_text(
        language="en",
        start_prompt="🎙️ Speak",
        stop_prompt="⏹️ Stop",
        just_once=True,
        key=f"stt_{idx}"
    )

    # Put spoken answer into the text box
    if spoken_text:
        st.session_state[f"answer_{idx}"] = spoken_text

    answer = st.text_area(
        "Type or edit your answer",
        key=f"answer_{idx}",
        height=170,
        placeholder="Type your answer or click 🎙️ Speak..."
    )
  

    if st.button("Submit answer", type="primary", use_container_width=True):
        if not answer.strip():
            st.warning("Please enter an answer before submitting.")
        else:
            with st.spinner("MockMate AI is evaluating your answer..."):
                feedback = ai_call(
                    f"""Target role: {st.session_state.role}
Question: {question}
Candidate answer:
{answer}
Evaluate the answer for this specific question."""
                )
            st.session_state.answers.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            })
            st.session_state.feedback.append(feedback)

            if idx == len(QUESTIONS) - 1:
                st.session_state.finished = True
                st.session_state.started = False
            else:
                st.session_state.q_index += 1
            st.rerun()

    if st.session_state.feedback:
        st.divider()
        st.subheader("Latest feedback")
        latest = st.session_state.feedback[-1]
        score = parse_score(latest)
        if score is not None:
            st.metric("Answer score", f"{score:.0f}/10")
        # st.markdown(latest)
        col1, col2 = st.columns([10,1])

        with col1:
            st.markdown(latest)

        with col2:
            if st.button("🔊", key=f"f_speak_{idx}"):
                speak_text(latest)
                # .....................change apply...........................

else:
    st.success("Interview complete 🎉")
    st.subheader("Your practice summary")

    scores = [parse_score(x) for x in st.session_state.feedback]
    scores = [x for x in scores if x is not None]
    if scores:
        st.metric("Average score", f"{sum(scores)/len(scores):.1f}/10")

    st.write(f"**Questions completed:** {len(st.session_state.answers)}")
    if st.session_state.start_time:
        mins = max(1, round((time.time() - st.session_state.start_time) / 60))
        st.write(f"**Approx. session time:** {mins} min")

    for i, (a, f) in enumerate(zip(st.session_state.answers, st.session_state.feedback), 1):
        with st.expander(f"Question {i}"):
            st.write("**Your answer:**")
            st.write(a["answer"])
            st.write("**AI feedback:**")
            st.markdown(f)

    st.divider()
    st.subheader("Quick product feedback")
    usefulness = st.slider("How useful was this practice?", 1, 5, 4)
    repeat_intent = st.selectbox("Would you use it again?", ["Definitely", "Probably", "Not sure", "No"])
    comments = st.text_area("What should we improve?")
    if st.button("Save feedback"):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": st.session_state.role,
            "questions_completed": len(st.session_state.answers),
            "usefulness_1_to_5": usefulness,
            "would_use_again": repeat_intent,
            "comments": comments,
        }
        out = Path("feedback.jsonl")
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success("Thanks — your feedback was recorded for this MVP session.")

    if st.button("🔄 Try another interview", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

st.caption("You are MockMate  AI • MVP built for product validation")
