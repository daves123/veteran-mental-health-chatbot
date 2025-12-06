import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1f4e78 0%, #2d6a9f 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'> About This Project</h1>
        <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>Veteran Mental Health RAG Chatbot</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Project Overview
st.header("📋 Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This **Retrieval-Augmented Generation (RAG) Chatbot** was developed as a Data Science 
    Masters capstone project to provide evidence-based information about veteran mental 
    health, with a particular focus on Post-Traumatic Stress Disorder (PTSD) and related 
    conditions.
    
    The system combines state-of-the-art natural language processing with carefully curated 
    health data sources to deliver accurate, contextual responses to questions about veteran 
    mental health challenges, treatments, and support resources.
    """)

with col2:
    st.info("""
    **Project Type:**  
    Data Science Masters Capstone
    
    **Focus Area:**  
    NLP, RAG Systems, Healthcare AI
    
    **Year:**  
    2024-2025
    """)

st.markdown("---")

# Key Features
st.header("✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Smart Retrieval
    - FAISS vector search
    - Semantic embeddings
    - Context-aware retrieval
    - Domain keyword boosting
    """)

with col2:
    st.markdown("""
    ### Rich Data Sources
    - BRFSS 2024
    - VA PTSD Repository
    - AHRQ Treatment Research
    - Medical terminology databases
    """)

with col3:
    st.markdown("""
    ### Specialized Answers
    - Definition detection
    - Symptom formatting
    - Treatment summaries
    - Gender-specific insights
    """)

st.markdown("---")

# Technology Stack
st.header("🛠️ Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Technologies")
    st.markdown("""
    - **Framework:** Python 3.10+
    - **Vector Database:** FAISS (Facebook AI Similarity Search)
    - **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
    - **Web Framework:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly
    """)

    st.subheader("Optional Components")
    st.markdown("""
    - **LLM Integration:** OpenAI GPT-3.5/4 (optional)
    - **ML Classification:** TensorFlow/Keras (optional)
    """)

with col2:
    st.subheader("RAG Architecture")
    st.code(
        """
    User Query
        ↓
    Query Encoding (Sentence-BERT)
        ↓
    FAISS Vector Search
        ↓
    Top-K Retrieval (k=5)
        ↓
    Context Assembly
        ↓
    Answer Generation
    (Extractive or LLM)
        ↓
    Response + Citations
    """,
        language="text",
    )

st.markdown("---")

# Data Sources
st.header("📚 Data Sources")

st.markdown("""
Our chatbot leverages multiple authoritative sources to ensure accuracy and comprehensiveness:
""")

# Create tabs for different source categories
tab1, tab2, tab3, tab4 = st.tabs(
    ["Government Surveys", "VA Research", "Treatment Studies", "Medical Terminology"]
)

with tab1:
    st.subheader("CDC BRFSS (Behavioral Risk Factor Surveillance System)")
    st.markdown("""
    - **2024 Survey Data:** 457,670 records from 49 states + DC + territories
    - **2023 Survey Data (not yet ingested):** Similar scale, independent sample  
    - **Veteran Subsets:** Separate processing for male and female veterans
    - **Topics Covered:** Mental health, chronic conditions, health behaviors, access to care
    
    **Key Note:** BRFSS is a cross-sectional survey with different people each year. 
    Records cannot be linked across years.
    
    🔗 [CDC BRFSS Website](https://www.cdc.gov/brfss/)
    """)

with tab2:
    st.subheader("VA PTSD Repository")
    st.markdown("""
    - **Study Characteristics Database:** Comprehensive PTSD research metadata
    - **Sample Demographics:** Veteran population characteristics
    - **Research Protocols:** Study designs and methodologies
    
    🔗 [PTSD Repository](https://ptsd-va.data.socrata.com/)
    """)

with tab3:
    st.subheader("AHRQ Evidence-Based Treatment Research")
    st.markdown("""
    - **Systematic Reviews:** Pharmacological and non-pharmacological PTSD treatments
    - **Comparative Effectiveness:** Evidence for different intervention types
    - **Treatment Characteristics:** Duration, format, delivery methods, completion rates
    
    🔗 [AHRQ PTSD Treatment Data](https://catalog.data.gov/dataset/ahrq-report-and-data-files-2023-pharmacological-and-nonpharmacological-treatments-for-post)
    """)

with tab4:
    st.subheader("Medical Abbreviations & Terminology")
    st.markdown("""
    - **VA PTSD Abbreviations Database:** Standardized medical terminology
    - **Clinical Assessment Tools:** MPSS, PSS-SR, PCL, CAPS
    - **Diagnostic Criteria:** DSM-5 symptom clusters
    
    🔗 [VA Medical Abbreviations](https://ptsd-va.data.socrata.com/PTSD-Repository/Abbreviations/46j5-9dq5/about_data)
    """)

st.markdown("---")
## Corpus Construction
st.header("🧠 Corpus Construction")
st.markdown("""
    - The final corpus contains **tens of thousands of knowledge chunks**
    - Each chunk includes:
    - Text content
    - Source attribution
    - Topic category
    - Gender metadata (male, female, or neutral)
    - Year

    **Corpus composition reflects responsible clinical design:**
    - Majority **gender-neutral** PTSD definitions, diagnostics, and treatments
    - Substantial **male** and **female** veteran-specific BRFSS health data
    - Balanced integration to support accurate, inclusive responses

    The finalized dataset is saved as:

    ```text
    veteran_rag_corpus.csv
    """)

st.markdown("---")

# Methodology
st.header("🔬 Methodology")

st.subheader("Data Processing Pipeline")

st.markdown("""
1. **Data Collection & Cleaning**
   - Downloaded from authoritative sources
   - Standardized formats (CSV, JSON)
   - Removed duplicates and invalid entries
   - Handled missing values appropriately

2. **Chunking Strategy**
   - Semantic chunking based on content type
   - Preserved context boundaries
   - Metadata tagging (source, year, topic, gender)
   - ~10,000+ knowledge chunks created

3. **Embedding Generation**
   - Sentence-BERT transformer model (all-MiniLM-L6-v2)
   - 384-dimensional dense vectors
   - Normalized for cosine similarity
   - Batch processing for efficiency

4. **Index Construction**
    - FAISS IndexFlatIP (Inner Product for normalized vectors)
    - Single FAISS index over the current corpus (BRFSS 2024 + PTSD knowledge)
    - Metadata preservation for filtering
    - Optimized for fast retrieval
    - Design supports separate indices per year if future BRFSS years are added

5. **Query Processing**
   - Query type detection (definition, symptom, treatment)
   - Smart answer formatting based on query intent
   - Domain keyword boosting for mental health terms
   - Source attribution and citation
""")

st.markdown("---")

# Team
st.header("👥 Project Team")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Dave Singh
    **Data Science Masters Student**
    
    - RAG system architecture
    - Data processing pipeline
    - Vector embedding implementation
    - Frontend development
    
    📧 Contact: [Add email if desired]  
    🔗 LinkedIn: [Add link if desired]
    """)

with col2:
    st.markdown("""
    ### Nipu Quayum
    **Data Science Masters Student**
    
    - Data collection & curation
    - Answer formatting logic
    - Testing & validation
    - Documentation
    
    📧 Contact: [Add email if desired]  
    🔗 LinkedIn: [Add link if desired]
    """)

st.markdown("---")

# Acknowledgments
st.header("🙏 Acknowledgments")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Data Sources
    - **CDC** - BRFSS survey data
    - **VA National Center for PTSD** - Research repository
    - **AHRQ** - Evidence-based treatment research
    - **Data.gov** - Open government data platform
    """)

with col2:
    st.markdown("""
    ### Technologies
    - **HuggingFace** - Sentence transformers
    - **Facebook Research** - FAISS vector search
    - **Streamlit** - Web application framework
    - **OpenAI** - Optional LLM integration
    """)

st.markdown("---")

# Use Cases & Target Audience
st.header("Use Cases & Target Audience")

tab1, tab2, tab3 = st.tabs(
    ["Veterans & Families", "Healthcare Providers", "Researchers"]
)

with tab1:
    st.markdown("""
    ### For Veterans & Families
    
    **Primary Use Cases:**
    - Understanding PTSD symptoms and diagnosis
    - Learning about available treatment options
    - Finding support resources and crisis help
    - Comparing male vs. female veteran experiences
    - Accessing evidence-based health information
    
    **Example Questions:**
    - "What are common PTSD symptoms?"
    - "What treatments are available for combat-related stress?"
    - "How can I help my veteran family member?"
    """)

with tab2:
    st.markdown("""
    ### For Healthcare Providers
    
    **Primary Use Cases:**
    - Quick reference for PTSD symptoms and criteria
    - Evidence-based treatment options and effectiveness
    - Understanding veteran-specific mental health challenges
    - Gender differences in presentation and treatment
    - Medical terminology and abbreviations
    
    **Example Questions:**
    - "What is Cognitive Processing Therapy?"
    - "What are DSM-5 PTSD diagnostic criteria?"
    - "What treatments have the strongest evidence base?"
    """)

with tab3:
    st.markdown("""
    ### For Researchers & Students
    
    **Primary Use Cases:**
    - Exploring BRFSS methodology and design
    - Understanding RAG architecture for healthcare
    - Learning about cross-sectional vs. longitudinal data
    - Reviewing treatment research literature
    - Data architecture decision-making
    
    **Example Questions:**
    - "How does BRFSS sampling work?"
    - "Why weren't 2023 and 2024 data combined?"
    - "What is the RAG retrieval process?"
    """)

st.markdown("---")

# Limitations & Disclaimers
st.header("⚠️ Limitations & Disclaimers")

col1, col2 = st.columns(2)

with col1:
    st.warning("""
    ### Medical Disclaimer
    
    **This chatbot is for educational purposes only.**
    
    - NOT a substitute for professional medical advice
    - NOT for diagnosis or treatment decisions
    - NOT intended for emergency situations
    - Does NOT replace consultation with healthcare providers
    
    **In Crisis?** Contact the Veterans Crisis Line:
    - Call: **988** then press **1**
    - Text: **838255**
    - Chat: [VeteransCrisisLine.net](https://www.veteranscrisisline.net/)
    """)

with col2:
    st.info("""
    ### Technical Limitations
    
    **Data Constraints:**
    - Knowledge cutoff: BRFSS 2024 data
    - Limited to included data sources
    - Cannot answer about very recent developments
    
    **System Limitations:**
    - Extractive answers (not generative by default)
    - Limited to retrieved context
    - May not capture all nuances
    - Retrieval quality depends on query phrasing
    
    **Privacy:**
    - All data is anonymized and aggregated
    - No personal health information stored
    - Conversations are not saved between sessions
    """)

st.markdown("---")

# Future Enhancements
st.header("🚀 Future Enhancements")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Data Expansion
    - [ ] Add 2025 BRFSS data
    - [ ] Include NHANES data
    - [ ] Military OneSource resources
    - [ ] State-specific programs
    - [ ] International veteran data
    """)

with col2:
    st.markdown("""
    ### Features
    - [ ] Multi-language support
    - [ ] Voice interface
    - [ ] Visualization of trends
    - [ ] Personalized recommendations
    - [ ] Treatment finder tool
    """)

with col3:
    st.markdown("""
    ### Technical
    - [ ] Fine-tuned embeddings
    - [ ] Hybrid search (dense + sparse)
    - [ ] Query expansion
    - [ ] Re-ranking models
    - [ ] Streaming responses
    """)

st.markdown("---")

# Contact & Feedback
st.header("📬 Contact & Feedback")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    We welcome feedback, suggestions, and collaboration opportunities!
    
    **Ways to Reach Us:**
    - 📧 Email: [Add your email]
    - 💼 LinkedIn: [Add LinkedIn profiles]
    - 🐙 GitHub: [Add repository link if public]
    - 🎓 Institution: [Add university name]
    
    **For Technical Issues:**
    Please report bugs or technical issues with:
    - Detailed description of the problem
    - Steps to reproduce
    - Screenshots if applicable
    - Your browser and operating system
    """)

with col2:
    st.info("""
    **Quick Links:**
    
    📊 [Data Architecture](/Data_Architecture)
    
    📈 [System Statistics](/Statistics)
    
    🏠 [Back to Chat](/Home)
    
    📚 [CDC BRFSS](https://www.cdc.gov/brfss/)
    
    🎖️ [VA PTSD](https://www.ptsd.va.gov/)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
    <p style='margin: 0; font-size: 1.1rem;'><strong>Veteran Mental Health RAG Chatbot</strong></p>
    <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Data Science Masters Capstone Project • 2024-2025</p>
    <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Built with ❤️ for Veterans and Their Families</p>
    <p style='margin: 1rem 0 0 0; font-size: 0.85rem; color: #999;'>
        Dave Singh & Nipu Quayum<br>
        Powered by RAG Technology • Streamlit • FAISS • Sentence-Transformers
    </p>
</div>
""",
    unsafe_allow_html=True,
)
