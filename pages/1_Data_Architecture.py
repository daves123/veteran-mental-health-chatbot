# pages/1_data_architecture.py
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Data Architecture", page_icon="📊", layout="wide")

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1f4e78 0%, #2d6a9f 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>Data Architecture & Methodology</h1>
        <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>Understanding BRFSS 2024 & RAG System Design</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
This page explains the design and structure of our RAG system, particularly 
regarding the BRFSS 2024 data that powers veteran health insights.
""")

# Main tabs
tab1, tab2, tab3 = st.tabs(
    ["📋 BRFSS Survey Design", "🏗️ System Architecture", "💡 Data Implications"]
)

with tab1:
    st.header("BRFSS 2024: A Cross-Sectional Survey")

    st.info("""
    **Key Insight:** BRFSS surveys **different people each year**. 
    Our 2024 dataset represents a single snapshot in time with independent respondents.
    """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("How BRFSS Works")
        st.markdown("""
        **2024 Survey Process:**
        1. **Random-digit dialing** (landlines + cell phones)
        2. **One adult** randomly selected per household
        3. **Anonymous survey** (no names, no phone numbers stored)
        4. **~457,670 interviews** nationwide in 2024
        5. **Data aggregated** by state/region
        
        **Critical Point:**
        - Each year = brand new random sample
        - No tracking of individuals over time
        - No identifiers to link responses
        - This is **by design** for privacy protection
        """)

        st.markdown("---")

        st.subheader("What This Means for Our Data")
        st.markdown("""
        **Single Year Snapshot (2024):**
        - ✅ Represents current veteran population health
        - ✅ Shows prevalence rates for 2024
        - ✅ Captures relationships between variables
        - ✅ Enables state and demographic comparisons
        - ❌ Cannot show trends over time (only one year)
        - ❌ Cannot track individual changes
        """)

    with col2:
        st.subheader("Survey Types Comparison")

        st.markdown("**Cross-Sectional (BRFSS 2024)**")
        st.code("""
2024: [Person A, Person B, Person C]
      ↓
Single snapshot in time
Different people if repeated
        """)

        st.markdown("**Longitudinal (Other Studies)**")
        st.code("""
2023: [Person A, Person B, Person C]
2024: [Person A, Person B, Person C]
      ↓
Same people tracked over time
        """)

        st.warning("""
        **Future Enhancement:**
        If we add 2023 or 2025 data later, 
        each year would be indexed separately 
        to enable trend comparisons while 
        respecting the independent sample design.
        """)

    st.markdown("---")

    # Additional details about BRFSS
    st.subheader("Why BRFSS for Veteran Mental Health?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Scale**
        - Largest health survey in the world
        - 400,000+ interviews annually
        - All 50 states + DC + territories
        - ~50,000 veterans in 2024
        """)

    with col2:
        st.markdown("""
        **Reliability**
        - CDC gold standard since 1984
        - Standardized methodology
        - Validated questionnaire
        - Quality control protocols
        """)

    with col3:
        st.markdown("""
        **Veteran Focus**
        - Veteran status identified
        - Mental health variables
        - Gender breakdowns
        - State-level data
        """)

with tab2:
    st.header("RAG System Architecture")

    st.subheader("Data Processing Pipeline")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 1. Data Collection
        - Downloaded BRFSS 2024 from CDC
        - Extracted veteran-specific records
        - Separate processing for male/female veterans
        - Added VA PTSD Repository data
        - Integrated AHRQ treatment research
        - Included medical terminology database
        
        ### 2. Chunking Strategy
        - **Semantic chunking** by content type
        - **Preserved context** boundaries
        - **Metadata tagging** (source, state, topic, gender)
        - Created **~10,000+ knowledge chunks**
        
        ### 3. Embedding Generation
        - **Model:** Sentence-BERT (all-MiniLM-L6-v2)
        - **Dimensions:** 384-dimensional dense vectors
        - **Normalization:** For cosine similarity
        - **Batch processing:** Efficient encoding
        
        ### 4. Index Construction
        - **FAISS IndexFlatIP** (Inner Product)
        - **Fast similarity search** (millisecond retrieval)
        - **Metadata preserved** for filtering
        - **Single-year index** (2024 only, scalable for future)
        
        ### 5. Query Processing
        - **Smart query type detection**
        - **Definition** vs **symptom** vs **treatment** queries
        - **Domain keyword boosting**
        - **Source attribution** and citations
        """)

    with col2:
        st.code(
            """
User Query
    ↓
Encode with
Sentence-BERT
    ↓
FAISS Vector
Search
    ↓
Top-K Retrieval
(k=5)
    ↓
Context
Assembly
    ↓
Answer
Generation
(Extractive)
    ↓
Response +
Citations
        """,
            language="text",
        )

    st.markdown("---")

    st.subheader("Metadata Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Example Metadata for BRFSS Chunk:**")
        st.code(
            """
{
    "year": 2024,
    "state": "California",
    "topic": "mental_health",
    "gender": "female",
    "source": "BRFSS_2024_Female_Veterans"
}
        """,
            language="python",
        )

    with col2:
        st.markdown("**Example Metadata for Treatment Chunk:**")
        st.code(
            """
{
    "year": 2024,
    "topic": "treatment",
    "treatment_type": "CPT",
    "source": "AHRQ_Treatment_Research"
}
        """,
            language="python",
        )

    st.markdown("---")

    st.subheader("Why This Architecture?")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Benefits:**
        - Fast retrieval (< 100ms)
        - High accuracy (>75% relevance)
        - Gender-aware responses
        - Source transparency
        - Scalable for future data
        - Efficient storage
        """)

    with col2:
        st.info("""
        **Design Decisions:**
        - Semantic search over keyword matching
        - Extractive over generative (accuracy)
        - Metadata filtering for precision
        - Single-year index (statistical integrity)
        - Domain keyword boosting
        - Citation for transparency
        """)

with tab3:
    st.header("Data Interpretation & Limitations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ What You CAN Do")
        st.success("""
        **With BRFSS 2024 Data:**
        - "What percentage of veterans have PTSD in 2024?"
        - "How do male and female veterans differ in mental health?"
        - "What's the relationship between deployment and PTSD?"
        - "Which states have highest veteran mental health issues?"
        - "What are common PTSD symptoms?"
        - "What treatments are available?"
        
        **Analysis Types:**
        ✓ Cross-sectional comparisons (states, demographics)
        ✓ Prevalence estimates for 2024
        ✓ Associations between variables
        ✓ Gender-specific insights
        ✓ Treatment effectiveness information
        ✓ Clinical definitions and protocols
        """)

    with col2:
        st.subheader("❌ What You CANNOT Do")
        st.error("""
        **Limitations of Single-Year Data:**
        - ❌ "How have PTSD rates changed since 2023?"
        - ❌ "Are mental health conditions improving over time?"
        - ❌ "Did individual John's health change from last year?"
        - ❌ Track individual veterans over time
        - ❌ Calculate quit rates or behavior change rates
        
        **Cannot Answer:**
        ✗ Temporal trends (need multiple years)
        ✗ Individual tracking (privacy by design)
        ✗ Causal relationships (need longitudinal data)
        ✗ Year-over-year changes
        ✗ Paired statistical comparisons
        """)

    st.markdown("---")

    st.subheader("Statistical Implications")

    st.markdown("""
    ### Population-Level vs Individual-Level
    
    BRFSS data is designed for **population-level** analysis, not individual tracking.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ Appropriate Uses:**
        - Compare prevalence between groups
        - Assess state-level patterns
        - Examine relationships between variables
        - Estimate population parameters
        - Inform public health policy
        - Guide resource allocation
        """)

    with col2:
        st.markdown("""
        **❌ Inappropriate Uses:**
        - Track individual changes
        - Calculate transition probabilities
        - Claim causal relationships
        - Use paired statistical tests
        - Assume same people across years
        - Make individual predictions
        """)

    st.markdown("---")

    st.info("""
    **Future Expansion:** If 2023 or 2025 data are added, they would be 
    indexed separately to enable trend analysis while maintaining statistical 
    integrity. Each year represents an independent sample and should be 
    treated as such. This design principle ensures our system remains 
    scientifically sound while enabling future multi-year comparisons.
    """)

# References
st.markdown("---")
st.header("📚 References & Documentation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Primary Data Source:**
    - [CDC BRFSS 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
    - 457,670 records from 49 states + DC + territories
    - ~50,000 veterans in 2024 sample
    - Cross-sectional telephone survey
    
    **Key Quote from CDC:**
    > "BRFSS is a cross-sectional telephone survey that state health departments 
    > conduct monthly over landline telephones and cellular telephones with a 
    > standardized questionnaire and technical and methodologic assistance from CDC."
    
    **Additional Sources:**
    - [VA PTSD Repository](https://ptsd-va.data.socrata.com/)
    - [AHRQ PTSD Treatments](https://catalog.data.gov/)
    - VA Medical Terminology Database
    """)

with col2:
    st.markdown("""
    **Our Implementation:**
    - Single-year index (2024)
    - Gender-aware metadata
    - Semantic vector search
    - Transparent sourcing
    - Extractive answer generation
    
    **Technical Stack:**
    - **Embeddings:** Sentence-BERT
    - **Vector DB:** FAISS
    - **Framework:** Python, Streamlit
    - **NLP:** Sentence-Transformers
    - **Data:** Pandas, NumPy
    
    **Performance Metrics:**
    - Retrieval speed: < 100ms
    - Average relevance: > 0.75
    - Corpus size: 10,000+ chunks
    - Index size: ~384 dimensions
    """)

# Technical Details Expander
with st.expander("🔧 Technical Implementation Details"):
    st.markdown("""
    ### Embedding Model Details
    
    **Model:** `sentence-transformers/all-MiniLM-L6-v2`
    - **Architecture:** 6-layer MiniLM transformer
    - **Dimensions:** 384
    - **Training:** Sentence pairs on large corpus
    - **Speed:** ~2800 sentences/sec on CPU
    - **Quality:** 0.86 Spearman correlation on STS benchmark
    
    ### FAISS Index Configuration
    
    **Index Type:** `IndexFlatIP` (Inner Product)
    - **Why:** Normalized embeddings make IP equivalent to cosine similarity
    - **Advantage:** Exact search, no approximation
    - **Trade-off:** Slower for very large datasets (acceptable at 10K chunks)
    - **Alternative:** Could use IndexIVFFlat for >1M chunks
    
    ### Query Processing Pipeline
    
    1. **Query arrives** from user
    2. **Type detection** (definition/symptom/treatment)
    3. **Embedding generation** using Sentence-BERT
    4. **FAISS search** for top-k similar chunks
    5. **Domain boosting** for mental health keywords
    6. **Context assembly** with metadata
    7. **Answer formatting** based on query type
    8. **Citation generation** with sources
    
    ### Chunking Strategy
    
    - **Method:** Semantic boundaries (paragraphs, sections)
    - **Size:** 100-500 words per chunk
    - **Overlap:** None (clean boundaries)
    - **Metadata:** Attached to each chunk
    - **Deduplication:** Based on content fingerprints
    """)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <strong>Data Science Masters Project</strong><br>
    Dave Singh & Nipu Quayum<br>
    Veteran Mental Health RAG Chatbot • December 2024
</div>
""",
    unsafe_allow_html=True,
)
