# app.py
import streamlit as st
import pandas as pd
import json
import fitz  # PyMuPDF
import docx
import requests
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import nltk
import warnings
import gc

from keybert import KeyBERT
from modules.keyword_extractor import extract_keywords
from modules.google_patents import search_google_patents_combined
from modules.similarity_engine import score_similarity
from modules.predict_ipc_or_cpc import predict_codes

# ========== Configurations ==========
st.set_page_config(page_title="🧠 Smart Invention Validator", layout="wide")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ========== Ensure NLTK ==========
for pkg in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

# ========== Reusable TogetherAI API Call ==========
def call_together(prompt, max_tokens=400, temperature=0.7, top_p=0.9):
    headers = {"Authorization": f"Bearer {st.secrets['together']['api_key']}"}
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post("https://api.together.xyz/inference", headers=headers, json=payload)
        return response.json().get("output", "") or response.json().get("choices", [{}])[0].get("text", "").strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ========== Styling ==========
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        flex-wrap: wrap;
    }
    h1, h2 {
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .element-container {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stProgress > div > div {
        background-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown("""
<div style='text-align:center;'>
    <h1 style='font-size:36px;'>🧠 Smart Invention Validator</h1>
    <p style='font-size:17px; color:#555;'>
        A streamlined AI assistant to validate your invention's novelty, extract key terms, find relevant patents, generate patent claims, and classify IPC/CPC codes.
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ========== Session Initialization ==========
def init_session():
    defaults = {
        "keywords": [],
        "patents": [],
        "abstract": "",
        "confirmed_abstract": False,
        "generated_claims_text": "",
        "predicted_ipc": [],
        "predicted_cpc": [],
        "similarity_clicked": False,
        "active_tab": 1  # Default navigation tab
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ========== Tabs ==========
tabs = st.tabs([
    "🏠 Home", "📄 Input", "🧩 Keywords", "🔎 Patent Search",
    "🧠 Similarity", "📜 Claims", "⚖️ Risk Score",
    "🏢 Assignees", "🔮 Classification Prediction", "🌳 IPC/CPC Tree", "📦 Summary & Export"
])

# ========== Tab 0: Homepage ==========
with tabs[0]:
    st.markdown("""
        <style>
            .overview-box {
                max-width: 900px;
                margin: auto;
                padding: 20px 30px;
                background-color: #f9f9f9;
                border-radius: 12px;
                font-size: 17px;
            }
            .overview-box h4 {
                margin-bottom: 10px;
                color: #222;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px 40px;
                margin-top: 20px;
            }
            .feature {
                background: white;
                padding: 12px 16px;
                border-radius: 10px;
                box-shadow: 0 0 5px rgba(0,0,0,0.05);
                transition: 0.3s;
            }
            .feature:hover {
                background-color: #fffaf0;
            }
            .emoji-label {
                font-weight: 600;
                color: #ff4b4b;
            }
        </style>

        <div class="home-container" style="text-align:center; padding-top: 30px;">
            <p style="font-size:18px; color:#444; max-width:800px; margin:auto;">
                Validate your invention using AI-powered tools for keyword extraction, patent discovery, risk scoring, and IPC/CPC classification — all accessible below.
            </p>
        </div>

        <div class="overview-box">
            <h4>🧭 App Features Overview</h4>
            <div class="feature-grid">
                <div class="feature"><span class="emoji-label">📄 Input:</span> Paste your invention abstract to begin validation.</div>
                <div class="feature"><span class="emoji-label">🧩 Keywords:</span> Extract technical and contextual keywords.</div>
                <div class="feature"><span class="emoji-label">🔍 Patent Search:</span> Retrieve relevant patents using keyword queries.</div>
                <div class="feature"><span class="emoji-label">🧠 Similarity:</span> Score your abstract against real patents.</div>
                <div class="feature"><span class="emoji-label">📜 Claims:</span> Generate one independent and two dependent claims.</div>
                <div class="feature"><span class="emoji-label">⚖️ Risk Score:</span> Estimate novelty/conflict risk.</div>
                <div class="feature"><span class="emoji-label">🏢 Assignees:</span> Identify top assignees owning similar IP.</div>
                <div class="feature"><span class="emoji-label">🔠 Classification Prediction:</span> Predict IPC/CPC codes using ML.</div>
                <div class="feature"><span class="emoji-label">🌲 IPC/CPC Tree:</span> View classification structure hierarchically.</div>
                <div class="feature"><span class="emoji-label">📦 Summary & Export:</span> Download results and final summary.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ========== Tab 1: Input ==========
with tabs[1]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>✍️ Invention Abstract Input</h2>
            <p style='color:gray;'>Paste your invention abstract below (minimum 200 characters or 40 words)</p>
        </div>
    """, unsafe_allow_html=True)

    abstract_input = st.text_area(
        "Enter Abstract",
        value=st.session_state["abstract"],
        height=250,
        label_visibility="collapsed",
        key="abstract_input"
    )

    submitted = st.button("📨 Submit Abstract", key="submit_abstract_btn")

    abstract_text = abstract_input.strip()
    word_count = len(abstract_text.split())
    char_count = len(abstract_text)
    is_short = word_count < 40 or char_count < 200
    is_generic = abstract_text.lower().endswith(("in general", "for a consumer", "or the like"))

    st.markdown(f"📉 **Abstract Word Count:** {word_count} words")

    if submitted:
        if not abstract_text:
            st.warning("⚠️ Text field is empty. Please enter your invention abstract.")
        elif is_short or is_generic:
            st.info("🧠 Your abstract seems short or vague. Want help improving it?")
        else:
            st.session_state["abstract"] = abstract_text
            st.success("✅ Abstract submitted successfully. You may now proceed to the Keywords tab.")

    if (is_short or is_generic) and abstract_text:
        if st.button("💡 Improve Abstract using Together AI", key="improve_abstract_btn"):
            with st.spinner("Generating improved version with Together AI..."):
                prompt = f"Improve this short patent abstract to be more detailed and formal:\n{abstract_text}"
                improved_text = call_together(prompt, max_tokens=400)
                if improved_text:
                    st.markdown("### ✨ Suggested Improved Abstract")
                    st.text_area("Improved Version", value=improved_text, height=250, key="improved_area")
                    st.warning("📋 Please copy the improved abstract above and paste it manually into the input box.")
                else:
                    st.error("❌ Could not extract improved abstract.")

# ========== Tab 2: Keywords ==========
with tabs[2]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🧩 Keyword Extraction</h2>
            <p style='color:gray;'>Generate and refine key phrases from your invention abstract</p>
        </div>
    """, unsafe_allow_html=True)

    abstract = st.session_state.get("abstract", "").strip()

    if not abstract:
        st.warning("⚠️ Please submit your abstract in the previous tab first.")
    else:
        if st.button("✨ Generate Keywords", key="generate_keywords_btn"):
            try:
                kw_model = KeyBERT(model='all-MiniLM-L6-v2')
                raw_keywords = kw_model.extract_keywords(
                    abstract,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.7,
                    top_n=12
                )
                keywords = [kw[0] for kw in raw_keywords]
            except:
                keywords = []

            if not keywords:
                prompt = f"Extract 10 important technical keywords or phrases from this invention abstract:\n{abstract}"
                text = call_together(prompt, max_tokens=300)
                fallback_keywords = [kw.strip().strip('"') for kw in text.split(',') if len(kw.strip()) > 1]
                keywords = fallback_keywords[:12]
                st.info("🧠 Keywords generated using Together AI fallback.")

            st.session_state["keywords"] = keywords

        if st.session_state.get("keywords"):
            st.success("✅ Keywords Extracted")
            edited_keywords = st.multiselect(
                "✏️ Refine your keywords (add/remove/reorder as needed)",
                options=st.session_state["keywords"],
                default=st.session_state["keywords"],
                key="keyword_selector"
            )
            if edited_keywords:
                st.session_state["keywords"] = edited_keywords

            st.markdown("### 🔑 Finalized Keywords")
            st.markdown("""<div style='display:flex; flex-wrap:wrap; gap:10px;'>""", unsafe_allow_html=True)
            for kw in st.session_state["keywords"]:
                st.markdown(f"""
                    <div style='padding:6px 12px; background:#f0f0f0; border-radius:20px; font-size:15px;'>
                        {kw}
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ========== Tab 3: Patent Search ==========
with tabs[3]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🔍 Patent Search</h2>
            <p style='color:gray;'>Find similar patents using your extracted keywords</p>
        </div>
    """, unsafe_allow_html=True)

    keywords = st.session_state.get("keywords", [])
    if not keywords:
        st.warning("⚠️ No keywords found. Please generate them first in the Keywords tab.")
    else:
        default_kw_string = ", ".join(keywords)
        kw_input = st.text_input("✏️ Refine search keywords", default_kw_string, key="refined_keyword_input")
        refined_keywords = [k.strip() for k in kw_input.split(",") if k.strip()]

        if st.button("🔎 Run Patent Search", key="run_patent_search_btn"):
            with st.spinner("🔍 Searching for similar patents..."):
                results = search_google_patents_combined(refined_keywords)
                seen_ids = set()
                unique_results = []

                for pat in results:
                    unique_id = pat.get("publication_number") or pat.get("grant_number") or pat.get("title")
                    if unique_id and unique_id not in seen_ids:
                        seen_ids.add(unique_id)
                        unique_results.append(pat)

                if unique_results and st.session_state.get("abstract"):
                    top_matches = score_similarity(st.session_state["abstract"], unique_results)
                    st.session_state["patents"] = top_matches
                else:
                    st.session_state["patents"] = unique_results

        if keywords:
            st.markdown("### 💡 Smart Query Suggestions")
            smart_query = f'"{" OR ".join(keywords[:5])}" site:patents.google.com'
            st.code(smart_query)
            st.markdown(f"Use this query in [Google Patents](https://patents.google.com) to explore results.")

        if st.session_state.get("patents"):
            st.success(f"✅ {len(st.session_state['patents'])} unique patents found")
            for pat in st.session_state["patents"]:
                title = pat.get("title", "Untitled Patent")
                patent_number = pat.get("patent_number", "N/A")
                similarity = int(pat.get("similarity", 0) * 100)
                assignee = pat.get("assignee", "Unknown")
                abstract = pat.get("abstract", "-")
                claim = pat.get("claim", "-")

                for kw in refined_keywords:
                    if len(kw) > 2:
                        abstract = abstract.replace(kw, f"<mark>{kw}</mark>")

                with st.expander(f"{title} ({patent_number}) — 🧠 Similarity: {similarity}%"):
                    st.markdown(f"📄 **Abstract:** <br><span style='line-height:1.6'>{abstract}</span>", unsafe_allow_html=True)
                    st.markdown(f"📜 **Claim:** {claim}")
                    st.markdown(f"📅 **Publication Date:** {pat.get('publication_date', '-')}")
                    st.markdown(f"🏢 **Assignee:** {assignee}")
                    st.markdown(f"🌐 **Source:** {pat.get('source', '-')}")
                    if patent_number != "N/A":
                        st.markdown(f"[🔗 View Full Patent](https://patents.google.com/patent/{patent_number})")

# ========== Tab 4: Similarity Analysis ==========
with tabs[4]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🧠 Similarity Analysis</h2>
            <p style='color:gray;'>Score and rank patents based on abstract similarity</p>
        </div>
    """, unsafe_allow_html=True)

    abstract = st.session_state.get("abstract", "").strip()
    patent_list = st.session_state.get("patents", [])

    if not abstract or not patent_list:
        st.warning("⚠️ Please submit an abstract and run the patent search first.")
    else:
        if st.button("🔬 Compute Semantic Similarity", key="compute_similarity_btn"):
            with st.spinner("Scoring similarity using embedding model..."):
                top_matches = score_similarity(abstract, patent_list)
                st.session_state["patents"] = top_matches
                st.session_state["similarity_clicked"] = True

        if st.session_state["similarity_clicked"]:
            st.success("✅ Patents scored by semantic similarity")
            st.markdown("### 🔝 Top Similar Matches")

            top_patents = sorted(st.session_state["patents"], key=lambda x: -x.get("similarity", 0))[:5]

            for pat in top_patents:
                similarity = int(pat.get("similarity", 0) * 100)
                title = pat.get("title", "Untitled")
                patent_number = pat.get("patent_number", "N/A")

                # Label
                if similarity >= 85:
                    label = "🟢 High"
                elif similarity >= 65:
                    label = "🟡 Medium"
                else:
                    label = "⚪ Low"

                with st.expander(f"{title} — {label} Similarity: {similarity}%"):
                    st.markdown(f"📄 **Abstract:** {pat.get('abstract', '-')}")
                    st.markdown(f"📜 **Claim:** {pat.get('claim', '-')}")
                    st.markdown(f"📅 **Publication Date:** {pat.get('publication_date', '-')}")
                    st.markdown(f"🏢 **Assignee:** {pat.get('assignee', '-')}")
                    st.markdown(f"🌐 **Source:** {pat.get('source', '-')}")
                    if patent_number != "N/A":
                        st.markdown(f"[🔗 View Full Patent](https://patents.google.com/patent/{patent_number})")

            st.markdown("### 📊 Similarity Score Comparison")
            chart_data = pd.DataFrame({
                "Patent Title": [p.get("title", "Untitled")[:40] for p in top_patents],
                "Similarity (%)": [int(p.get("similarity", 0) * 100) for p in top_patents]
            })

            fig, ax = plt.subplots()
            chart_data.plot(kind="barh", x="Patent Title", y="Similarity (%)", ax=ax, legend=False, color="#76b5c5")
            plt.gca().invert_yaxis()
            ax.set_xlabel("Similarity (%)")
            st.pyplot(fig)

# ========== Tab 5: Claim Generation ==========
with tabs[5]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>📜 Claim Generator</h2>
            <p style='color:gray;'>Auto-generate structured patent claims from your abstract</p>
        </div>
    """, unsafe_allow_html=True)

    abstract = st.session_state.get("abstract", "").strip()

    def generate_claims():
        prompt = f"""
You are a professional patent claim writer. Based on the invention abstract below, write:

- One **independent claim** using a clear preamble (e.g., “A method for...”, “A system comprising...”), followed by at least 3 structural or functional steps.
- Two **dependent claims** that build logically upon the independent claim and add specific technical details.

Avoid leaving any claim open-ended. Each sentence should be complete and self-contained.

Format:
1. [Independent Claim]
2. [Dependent Claim 1]
3. [Dependent Claim 2]

Abstract:
{abstract}

Claims:
"""
        return call_together(prompt, max_tokens=900)

    if st.button("⚙️ Generate Claims", key="generate_claims_btn") or st.button("🔁 Regenerate Claims", key="regenerate_claims_btn"):
        if len(abstract.split()) < 20:
            st.warning("⚠️ The abstract may be too short or vague for reliable claim generation.")
        else:
            with st.spinner("Generating claims using Together AI..."):
                claim_output = generate_claims()
                st.session_state["generated_claims_text"] = claim_output

    claims_raw = st.session_state.get("generated_claims_text", "")

    if claims_raw:
        if "1." in claims_raw:
            st.success("✅ Structured Claims Generated")
            claims = claims_raw.strip().split("\n")
            for line in claims:
                if line.startswith("1."):
                    st.markdown("### 🟢 Independent Claim")
                    st.markdown(f"<div style='padding:10px;border:1px solid #ccc;border-radius:6px;background:#f9f9f9'>{line}</div>", unsafe_allow_html=True)
                elif line.startswith("2.") or line.startswith("3."):
                    if line.startswith("2."):
                        st.markdown("### 🔵 Dependent Claims")
                    st.markdown(f"<div style='padding:8px 15px;margin-bottom:6px;border-left:4px solid #007ACC;background:#f0f7ff'>{line}</div>", unsafe_allow_html=True)

            st.markdown("### ✍️ Manually Edit or Copy Full Claim Set")
            st.text_area("📄 Full Claims Output", value=claims_raw, height=250, key="manual_claim_edit")
        else:
            st.error("❗ Claim output is incomplete or incorrectly formatted.")
            st.text_area("📄 Raw Output", value=claims_raw, height=250, key="raw_claim_error")

# ========== Tab 6: Risk Score ==========
with tabs[6]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>⚖️ Risk Assessment</h2>
            <p style='color:gray;'>Assess novelty and overlap risk based on AI and similarity analysis</p>
        </div>
    """, unsafe_allow_html=True)

    abstract = st.session_state.get("abstract", "")
    patents = st.session_state.get("patents", [])
    keywords = st.session_state.get("keywords", [])
    ipc_predictions = predict_codes(abstract, model_type="ipc") if abstract else []

    if st.button("📊 Run Risk Analysis", key="run_risk_analysis_btn"):
        word_count = len(abstract.split())
        num_matches = len(patents)
        similarities = [pat.get("similarity", 0) for pat in patents if "similarity" in pat]
        max_sim = max(similarities) if similarities else 0

        # ===== RISK SCORING =====
        risk_score = 0
        reasons = []

        if max_sim >= 0.75:
            risk_score += 4
            reasons.append("🔴 High similarity to existing patents")
        elif 0.50 <= max_sim < 0.75:
            risk_score += 3
            reasons.append("🟠 Moderate similarity to prior art")
        elif 0.35 <= max_sim < 0.50:
            risk_score += 2
            reasons.append("🟡 Some overlap with existing patents")

        if word_count < 20:
            risk_score += 2
            reasons.append("✏️ Abstract too short (<20 words)")

        if not keywords:
            risk_score += 1
            reasons.append("❌ No keywords extracted")

        if not ipc_predictions:
            risk_score += 1
            reasons.append("🧠 No IPC classification predicted")

        if num_matches > 5:
            risk_score += 1
            reasons.append("📄 Multiple similar patents found")

        # ===== RISK LABEL =====
        if risk_score <= 2:
            badge = "🟢 Low Risk — Likely Novel and Well Described."
            color = "success"
        elif risk_score <= 5:
            badge = "🟡 Medium Risk — Potential overlap or vague coverage."
            color = "warning"
        else:
            badge = "🔴 High Risk — Review abstract or claims thoroughly."
            color = "error"

        # ===== OUTPUT =====
        st.markdown(f"### 🔍 Max Similarity Score: **{int(max_sim * 100)}%**")
        st.markdown("### 📄 Risk Insights")
        for reason in reasons:
            st.markdown(f"- {reason}")

        st.markdown("### 📊 Final Risk Score")
        st.progress(min(risk_score / 8, 1.0))
        getattr(st, color)(badge)

        # ===== EXPORT REPORT =====
        report = f"""SMART INVENTION VALIDATOR — RISK REPORT

Abstract Word Count: {word_count}
Similar Patents Found: {num_matches}
Max Similarity Score: {int(max_sim * 100)}%
Keywords Extracted: {len(keywords)}
IPC Codes Predicted: {len(ipc_predictions)}
Final Risk Score (0–8): {risk_score}

Interpretation:
{badge}

Risk Insights:
{chr(10).join(reasons)}
"""
        st.download_button("📥 Download Risk Report", report, file_name="risk_score_report.txt", key="risk_download")

# ========== Tab 7: Assignees ==========
with tabs[7]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🏢 Assignee Landscape</h2>
            <p style='color:gray;'>Visualize top organizations holding similar patents</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("📊 Show Top Assignees", key="show_assignees_btn"):
        patents = st.session_state.get("patents", [])
        assignees = [pat.get("assignee", "Unknown") for pat in patents if pat.get("assignee")]

        from collections import Counter
        assignee_counts = Counter(assignees)

        if assignee_counts:
            df = pd.DataFrame(assignee_counts.items(), columns=["Assignee", "Count"]).sort_values("Count", ascending=False)
            st.markdown("### 📈 Top 10 Assignees")
            st.dataframe(df.head(10), use_container_width=True)

            # Chart
            fig, ax = plt.subplots(figsize=(8, 5))
            df.head(10).plot(kind="barh", x="Assignee", y="Count", ax=ax, legend=False, color="#86c5ff")
            ax.set_xlabel("Number of Patents")
            ax.set_title("Top Assignees in Search Results")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            st.download_button("📥 Download Assignee List", df.to_csv(index=False), file_name="top_assignees.csv", key="assignee_csv_download")
        else:
            st.warning("⚠️ No assignee data found. Run a patent search first.")

# ========== Tab 8: IPC/CPC Code Prediction ==========
with tabs[8]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🔮 Classification Prediction</h2>
            <p style='color:gray;'>Predict classification codes from your invention abstract</p>
        </div>
    """, unsafe_allow_html=True)

    def load_descriptions(code_type):
        try:
            with open(f"data/{code_type}_code_descriptions.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {code_type.upper()} descriptions: {str(e)}")
            return {}

    def get_definition_with_fallback(code, defs):
        clean_code = code.replace("/", "").replace(" ", "").strip()
        variants = [
            code, code.replace("/", ""), code.split("/")[0],
            code[:5].replace("/", ""), code[:4].replace("/", ""),
            code[:3].replace("/", ""), code[0]
        ]
        for variant in variants:
            if variant in defs:
                return defs[variant]
        for key in defs:
            if code.startswith(key) or clean_code.startswith(key):
                return defs[key]
        return {"title": "N/A"}

    abstract = st.session_state.get("abstract", "").strip()

    if abstract:
        if st.button("🧠 Predict IPC/CPC Codes", key="predict_codes_btn"):
            ipc_results = predict_codes(abstract, model_type="ipc", top_k=4)
            cpc_results = predict_codes(abstract, model_type="cpc", top_k=4)
            st.session_state["predicted_ipc"] = ipc_results
            st.session_state["predicted_cpc"] = cpc_results

        ipc_defs = load_descriptions("ipc")
        cpc_defs = load_descriptions("cpc")

        if st.session_state.get("predicted_ipc"):
            st.success("✅ IPC Predicted Codes")
            for item in st.session_state["predicted_ipc"]:
                defn = get_definition_with_fallback(item["code"], ipc_defs)
                st.markdown(f"**{item['code']}** — {defn.get('title', 'N/A')}")

        if st.session_state.get("predicted_cpc"):
            st.success("✅ CPC Predicted Codes")
            for item in st.session_state["predicted_cpc"]:
                defn = get_definition_with_fallback(item["code"], cpc_defs)
                st.markdown(f"**{item['code']}** — {defn.get('title', 'N/A')}")
    else:
        st.warning("⚠️ Please enter your abstract first to predict classification.")

# ========== Tab 9: IPC/CPC Tree ==========
with tabs[9]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>🌳 IPC/CPC Hierarchy Tree</h2>
            <p style='color:gray;'>Visual representation of classification structure and co-occurrence</p>
        </div>
    """, unsafe_allow_html=True)

    model_choice = st.radio("🔄 Choose model type:", ["ipc", "cpc"], horizontal=True, key="tree_model_choice")

    if st.button("🌐 Generate Tree & Matrix", key="generate_tree_btn"):
        abstract = st.session_state.get("abstract", "")
        if not abstract:
            st.warning("⚠️ Please submit a valid abstract first.")
        else:
            hierarchy_data = predict_codes(abstract, model_type=model_choice)

            if not hierarchy_data:
                st.error("❌ No codes returned from the model.")
            else:
                max_limit = 50 if model_choice == "cpc" else 100
                if len(hierarchy_data) > max_limit:
                    st.info(f"⚠️ Displaying only top {max_limit} codes to avoid memory overload.")
                    hierarchy_data = hierarchy_data[:max_limit]

                # Tree
                st.markdown("### 📚 Classification Tree")
                G = nx.DiGraph()
                for res in hierarchy_data:
                    part = res["code"].strip()
                    section = part[:1]
                    subclass = part.split("/")[0]
                    G.add_edge(section, subclass)
                    G.add_edge(subclass, part)

                fig_tree, ax_tree = plt.subplots(figsize=(8, 5))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, arrows=False, node_size=1800, node_color="#cce5ff", font_size=10, ax=ax_tree)
                st.pyplot(fig_tree)
                del G, fig_tree, ax_tree; gc.collect()

                # Co-occurrence
                st.markdown("### 🔗 Co-Occurrence Matrix")
                codes = [res["code"] for res in hierarchy_data]
                co_matrix = pd.DataFrame(0, index=codes, columns=codes)
                for i in range(len(codes)):
                    for j in range(len(codes)):
                        if i != j:
                            co_matrix.loc[codes[i], codes[j]] += 1

                fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
                cax = ax_heat.matshow(co_matrix.values, cmap="Blues")
                ax_heat.set_xticks(range(len(codes)))
                ax_heat.set_xticklabels(codes, rotation=90)
                ax_heat.set_yticks(range(len(codes)))
                ax_heat.set_yticklabels(codes)
                fig_heat.colorbar(cax)
                st.pyplot(fig_heat)
                del fig_heat, ax_heat, co_matrix; gc.collect()
    else:
        st.info("Click the button above to generate the IPC/CPC structure and matrix.")

# ========== Tab 10: Summary & Export ==========
with tabs[10]:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#333;'>📦 Summary & Export</h2>
            <p style='color:gray;'>View a summary of all steps and download results</p>
        </div>
    """, unsafe_allow_html=True)

    abstract = st.session_state.get("abstract", "")
    keywords = st.session_state.get("keywords", [])
    patents = st.session_state.get("patents", [])
    ipc_codes = st.session_state.get("predicted_ipc", [])
    cpc_codes = st.session_state.get("predicted_cpc", [])

    st.subheader("📄 Abstract")
    st.write(abstract or "Not available")

    st.subheader("🧩 Keywords")
    st.write(", ".join(keywords) if keywords else "Not available")

    st.subheader("🔍 Top Patent Results")
    if patents:
        for i, pat in enumerate(patents[:3], 1):
            title = pat.get('title', '-')[:100]
            similarity = round(pat.get('similarity', 0) * 100)
            st.markdown(f"**{i}. {title}...**")
            st.markdown(f"🧠 Similarity: {similarity}%")
    else:
        st.write("No patents found.")

    st.subheader("🔮 IPC / CPC Codes")
    if ipc_codes:
        st.markdown("**IPC Codes:**")
        for code in ipc_codes:
            st.markdown(f"- {code['code']}: {code.get('title', 'N/A')}")
    if cpc_codes:
        st.markdown("**CPC Codes:**")
        for code in cpc_codes:
            st.markdown(f"- {code['code']}: {code.get('title', 'N/A')}")

    # Export
    export_text = f"""SMART INVENTION VALIDATOR — SUMMARY EXPORT

ABSTRACT:
{abstract}

KEYWORDS:
{', '.join(keywords)}

TOP PATENTS:
"""
    for i, pat in enumerate(patents[:5], 1):
        export_text += f"{i}. {pat.get('title', '-')[:100]}... (Similarity: {round(pat.get('similarity', 0)*100)}%)\n"
    export_text += "\nIPC CODES:\n" + "\n".join([f"{c['code']}: {c.get('title', 'N/A')}" for c in ipc_codes])
    export_text += "\nCPC CODES:\n" + "\n".join([f"{c['code']}: {c.get('title', 'N/A')}" for c in cpc_codes])

    st.download_button("📥 Download Full Summary", export_text, file_name="invention_summary.txt", key="download_summary_btn")

# ========== Footer ==========
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px; color:gray;'>
Built with ❤️ using Streamlit • Smart Invention Validator © 2025
</p>
""", unsafe_allow_html=True)
