import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from groq import Groq

# Load environment variables (Local development fallback)
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Product Comparison Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    .reportview-container { background: #f5f7f9; }
    .main .block-container { padding-top: 2rem; }
    div.stButton > button:first-child {
        background-color: #0066cc; color: white; border-radius: 6px; border: none;
    }
    div.stButton > button:first-child:hover { background-color: #0052a3; }
    .card {
        background: white; padding: 20px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #0066cc; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = {}  # {filename: structural_text}
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = []  # List of normalized dicts
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Secure API Key Retrieval
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-8b-8192"

if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key missing! Please set GROQ_API_KEY in your Streamlit Secrets or local .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- File Extraction Functions ---
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

# --- AI Core Data Extraction ---
def parse_product_with_llm(filename, raw_text):
    """Uses LLM to extract key technical attributes in a reliable JSON structure."""
    prompt = f"""
    Analyze the following product specification document text and extract architectural specifications into a clean JSON structure.
    If an item is missing or not declared, fill it as null. Do not hallucinate or guess.

    Text Context:
    ---
    {raw_text}
    ---

    Respond ONLY with a valid JSON object following exactly this schema:
    {{
        "product_name": "Standard String (Clean name of the product)",
        "price": 0.0,
        "key_features": ["feature 1", "feature 2"],
        "technical_specifications": {{
            "spec_name_1": "spec_value_1",
            "spec_name_2": "spec_value_2"
        }},
        "advantages": ["advantage 1", "advantage 2"],
        "disadvantages": ["disadvantage 1", "disadvantage 2"]
    }}
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices.message.content)
    except Exception as e:
        st.error(f"LLM extraction error for file {filename}: {e}")
        return None

# --- AI Core Recommendation System ---
def generate_recommendation_report(products, criteria):
    """Generates structured cross-evaluation, mapping options against hard constraints."""
    prompt = f"""
    You are an expert product advisor. Compare the following structured products against the explicit user preferences and provide an actionable recommendation report.

    User Constraints:
    - Maximum Budget: {criteria['budget']}
    - Must-Have Features: {criteria['must_have']}
    - Optional/Preferred Features: {criteria['optional']}
    - Intended Use Case: {criteria['usage']}

    Products Pool Datasets:
    {json.dumps(products, indent=2)}

    Instructions:
    1. Validate items strictly against hard constraints (Budget and Must-Have Features). Mark any violating item as non-compliant.
    2. Rank the choices based on technical suitability for the stated usage scenario.
    3. Detail trade-offs comprehensively.

    Format your output cleanly in Markdown with sections:
    ### 🏆 Best Choice Recommendation
    ### 📊 Critical Trade-off Analysis
    ### ⚠️ Constraint Deviations (If applicable)
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.3
        )
        return response.choices.message.content
    except Exception as e:
        return f"Could not generate AI recommendation due to API error: {e}"

# --- Layout: Sidebar ---
with st.sidebar:
    st.title("⚙️ Operations Panel")
    st.subheader("1. File Ingestion Pipeline")
    uploaded_files = st.file_uploader(
        "Upload spec documents", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.raw_docs:
                with st.spinner(f"Reading {f.name}..."):
                    if f.name.endswith(".pdf"):
                        txt = extract_text_from_pdf(f)
                    elif f.name.endswith(".docx"):
                        txt = extract_text_from_docx(f)
                    else:
                        txt = extract_text_from_txt(f)
                    
                    if txt.strip():
                        st.session_state.raw_docs[f.name] = txt
        
        if st.button("⚡ Execute AI Processing"):
            extracted = []
            progress_bar = st.progress(0)
            for idx, (name, raw_content) in enumerate(st.session_state.raw_docs.items()):
                with st.spinner(f"Extracting structured specs from {name}..."):
                    parsed_json = parse_product_with_llm(name, raw_content)
                    if parsed_json:
                        if not parsed_json.get("product_name"):
                            parsed_json["product_name"] = name.split(".")
                        extracted.append(parsed_json)
                progress_bar.progress((idx + 1) / len(st.session_state.raw_docs))
            st.session_state.extracted_data = extracted
            st.success("Data Extraction Complete!")

# --- Main App Core ---
st.title("📊 AI-Powered Product Comparison & Recommendation Tool")
st.write("Upload raw datasheets, configure parameters, and get comprehensive trade-off analytics instantly.")

# User Requirements Input Block
st.subheader("🎯 Target Profile & Constraints Configuration")
with st.form("user_requirements_form"):
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Maximum Budget ($)", min_value=0.0, value=1500.0, step=50.0)
        must_have = st.text_input("Must-Have Features (comma-separated)", "Battery, Wireless")
    with col2:
        usage = st.text_area("Primary Use Case / Intended Workload Description", "Professional remote video rendering work.")
        optional = st.text_input("Optional/Preferred Features (comma-separated)", "OLED screen, Lightweight")
        
    submit_reqs = st.form_submit_button("Save & Recalculate Scoring Matrix")

# Package current criteria
criteria = {
    "budget": budget,
    "must_have": [x.strip() for x in must_have.split(",") if x.strip()],
    "optional": [x.strip() for x in optional.split(",") if x.strip()],
    "usage": usage
}

# --- Analytical Views & Tabs Layout ---
if st.session_state.extracted_data:
    products = st.session_state.extracted_data
    
    scoring_matrix = []
    for p in products:
        p_name = p.get("product_name", "Unknown Product")
        p_price = float(p.get("price") or 0.0)
        p_features = [f.lower() for f in p.get("key_features", [])]
        
        # Hard compliance tracking
        budget_ok = p_price <= criteria["budget"]
        missing_must_haves = [m for m in criteria["must_have"] if m.lower() not in "".join(p_features)]
        hard_compliance = budget_ok and (len(missing_must_haves) == 0)
        
        # Match scoring
        mh_matches = sum(1 for m in criteria["must_have"] if m.lower() in "".join(p_features))
        opt_matches = sum(1 for o in criteria["optional"] if o.lower() in "".join(p_features))
        total_possible = len(criteria["must_have"]) + len(criteria["optional"])
        score = round(((mh_matches + opt_matches) / total_possible) * 100, 2) if total_possible > 0 else 100.0
        
        scoring_matrix.append({
            "Product": p_name,
            "Price ($)": p_price,
            "Budget Compliant": "✅" if budget_ok else "❌",
            "Features Match Score (%)": score,
            "Passes Hard Constraints": "🟢 Compliant" if hard_compliance else "🔴 Fails Requirements",
            "Advantages": ", ".join(p.get("advantages", [])[:3]),
            "Disadvantages": ", ".join(p.get("disadvantages", [])[:3])
        })
        
    df_metrics = pd.DataFrame(scoring_matrix)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Structured Comparison Table", 
        "📈 Visual Analytics Dashboard", 
        "💡 AI Recommendations & Trade-offs", 
        "💬 Interactive Assistant Context"
    ])

    # TAB 1: Structured Comparison Table
    with tab1:
        st.subheader("Cross-Product Specification Matrix")
        st.dataframe(
            df_metrics,
            column_config={
                "Features Match Score (%)": st.column_config.ProgressColumn(
                    "Features Match Score", min_value=0, max_value=100, format="%f%%"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("🔎 Expand Complete Extracted Technical Specs"):
            for p in products:
                st.markdown(f"**Product: {p.get('product_name')}**")
                st.json(p.get("technical_specifications", {}))

    # TAB 2: Visual Analytics Dashboard
    with tab2:
        st.subheader("Visual Structural Analytics")
        gcol1, gcol2 = st.columns(2)
        
        with gcol1:
            fig_price = px.bar(
                df_metrics, x="Product", y="Price ($)", 
                title="Price Distribution Comparison",
                color="Product", text_auto=True
            )
            fig_price.add_hline(y=criteria["budget"], line_dash="dash", line_color="red", annotation_text="Max Budget Target")
            st.plotly_chart(fig_price, use_container_width=True)
            
        with gcol2:
            fig_score = px.bar(
                df_metrics, x="Product", y="Features Match Score (%)",
                title="Weighted User Requirement Matching Scores",
                color="Passes Hard Constraints",
                color_discrete_map={"🟢 Compliant": "#2ca02c", "🔴 Fails Requirements": "#d62728"}
            )
            st.plotly_chart(fig_score, use_container_width=True)

    # TAB 3: AI Recommendations & Trade-offs
    with tab3:
        st.subheader("AI-Driven Constraint & Value Optimization Analysis")
        with st.spinner("AI evaluating ecosystem variables..."):
            report = generate_recommendation_report(products, criteria)
            st.markdown(report)

    # TAB 4: Interactive Assistant Context
    with tab4:
        st.subheader("Interactive Specialized Chat System")
        st.write("Ask focused queries about these specific models. The engine operates entirely inside the scope of your uploaded configurations.")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_query := st.chat_input("Ex: Which model delivers maximum value within budget limits?"):
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            chat_context_prompt = f"""
            You are a helpful product analytics chatbot. Answer the user query using ONLY the verified product context data provided below. Do not make up facts.
            If the answer cannot be confidently deduced from the data, explicitly state that the specification or detail is missing.

            User Configuration Criteria:
            {json.dumps(criteria, indent=2)}

            Verified Ingested Product Records:
            {json.dumps(products, indent=2)}

            Query: {user_query}
            """
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": chat_context_prompt}],
                    model=MODEL_NAME,
                    temperature=0.2
                )
                ai_response = response.choices.message.content
            except Exception as e:
                ai_response = f"Error consulting inference engine: {e}"
                
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
else:
    st.info("📥 Ingestion phase pending. Use the left Operations Panel to drop text/datasheets and begin comparison lifecycle.")
