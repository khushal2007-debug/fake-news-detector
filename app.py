import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from typer import prompt

load_dotenv()

#page config

st.set_page_config(
    page_title="Fake News Detector",
    page_icon=":newspaper:",
    layout="centered",
    initial_sidebar_state="auto",
)
# Initialize the Mistral AI model
llm=ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3,
)

#detecting fake news
def detect_fake_news(news_text):
     prompt = PromptTemplate(
        input_variables=["news"],
        template="""
        You are an expert fact-checker and journalist.
        
        Analyze this news article carefully:
        {news}
        
        Give your analysis in this EXACT format:
        
        VERDICT: [REAL / FAKE / MISLEADING]
        
        CONFIDENCE: [0-100%]
        
        REASONS:
        - [Reason 1]
        - [Reason 2]
        - [Reason 3]
        
        RED FLAGS:
        - [Any suspicious elements]
        
        VERDICT EXPLANATION:
        [2-3 lines explanation]
        
        ADVICE:
        [What reader should do]
        """
    )
     chain=prompt|llm
     result=chain.invoke({"news": news_text})
     return result.content

# Streamlit app

#header

st.header("Fake News Detector")
st.subheader("AI powered news verification tool")
st.markdown("----")

#info box

st.info("""
💡 **How it works:**
- Paste any news article or headline
- AI will analyze it
- Get verdict with confidence score
""")
#input

news_input = st.text_area(
    "📰 Paste News Article Here:",
    placeholder="Enter news headline or full article...",
    height=200
)

col1,col2,col3=st.columns([1,2,1])
with col2:
    analyze_btn=st.button(
         "🔍 Analyze News",
        use_container_width=True
    )
st.markdown("---")

#results

if analyze_btn:
    if news_input.strip():
        with st.spinner("🤖 AI analyzing news..."):
            result = detect_fake_news(news_input)
            # Verdict Color
        if "FAKE" in result:
            st.error("### ⚠️ VERDICT: FAKE NEWS DETECTED!")
        elif "MISLEADING" in result:
            st.warning("### ⚠️ VERDICT: MISLEADING NEWS!")
        else:
            st.success("### ✅ VERDICT: NEWS APPEARS REAL")

        st.markdown("### 📊 Detailed Analysis")
        st.markdown(result)

        st.markdown("---")
        st.caption("⚠️ AI can make mistakes. Always verify from multiple sources.")
    else:
        st.warning("⚠️ Please paste a news article first!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by **Khushal** | Fake News Detector AI")