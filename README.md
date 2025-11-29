# 🎖️ Veteran Mental Health RAG Chatbot

An evidence-based mental health information system for veterans, families, and healthcare providers using Retrieval-Augmented Generation (RAG) technology.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io/)

---

## Project Overview

This chatbot addresses a critical need in veteran mental healthcare by providing 24/7 access to evidence-based information about PTSD and related conditions. Built with a focus on female veterans—an underserved population in mental health research—the system draws from authoritative government and clinical sources to deliver accurate, cited responses.

**Personal Motivation:** This project stems from direct experience—my spouse is a veteran, and I've seen firsthand the challenges veterans face in accessing mental health information and care.

---

## Key Features

- **Evidence-Based Answers:** All responses grounded in authoritative sources with citations
- **Gender-Specific Analysis:** Dedicated focus on female veteran mental health challenges
- **Treatment Information:** Detailed descriptions of evidence-based PTSD treatments
- **Transparent Sources:** Clickable links to verify all data sources
- **Real-Time Retrieval:** Sub-second query response time after initial load
- **Professional Formatting:** Clean, readable answers with proper structure
- **System Transparency:** Statistics dashboard shows scale and data distribution

---

## System Specifications

| Metric | Value |
|--------|-------|
| **Knowledge Chunks** | 32,984 |
| **Data Sources** | 68 unique sources |
| **Indexed Vectors** | 32,984 (384-dimensional) |
| **Primary Dataset** | BRFSS 2024 (CDC) |
| **Retrieval Engine** | FAISS (Facebook AI Similarity Search) |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Response Time** | <1 second (after warm-up) |

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- 2GB RAM minimum
- ~3GB disk space for data and models

### Installation and RUN locally

```bash
# Clone the repository
git clone <repository-url>
cd veteran_mental_health_chatbot

# Install dependencies
pip install -r requirements-minimal.txt

# Run the chatbot
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First-Time Setup

On first launch, the system will:
1. Load the corpus (32,984 chunks)
2. Create embeddings (~3-5 minutes)
3. Build FAISS index
4. Save index to disk

Subsequent launches are near-instant (~5 seconds).

---

## Data Sources

All data comes from publicly available, authoritative sources:

| Source | Description | Link |
|--------|-------------|------|
| **BRFSS 2024** | CDC Behavioral Risk Factor Surveillance System | [View](https://www.cdc.gov/brfss/annual_data/annual_2024.html) |
| **Female Veteran Data** | Pre-cleaned BRFSS 2024 subset (female veterans) | [View](https://www.cdc.gov/brfss/annual_data/annual_2024.html) |
| **Male Veteran Data** | Pre-cleaned BRFSS 2024 subset (male veterans) | [View](https://www.cdc.gov/brfss/annual_data/annual_2024.html) |
| **VA PTSD Repository** | Sample characteristics from VA research | [View](https://ptsd-va.data.socrata.com/PTSD-Repository/Study-Characteristics/npcj-egem/about_data) |
| **Medical Abbreviations** | VA PTSD terminology database | [View](https://ptsd-va.data.socrata.com/PTSD-Repository/Abbreviations/46j5-9dq5/about_data) |
| **AHRQ Treatment Research** | Systematic review of PTSD treatments | [View](https://catalog.data.gov/dataset/ahrq-report-and-data-files-2023-pharmacological-and-nonpharmacological-treatments-for-post) |

**For complete documentation:** See [DATA_SOURCES.md](DATA_SOURCES.md)

---

## Demo Queries

Try these queries to see the system in action:

### Symptoms & Diagnosis
```
What are common PTSD symptoms in veterans?
How is PTSD diagnosed?
```

### Treatments
```
What treatments are available for PTSD?
What is Cognitive Processing Therapy?
What medications are used for PTSD?
```

### Gender-Specific
```
What mental health challenges do female veterans face?
How can a female veteran get help for PTSD?
```

### Support & Resources
```
What support resources exist for veterans?
Where can veterans find mental health services?
```

---

## Project Structure

```
veteran_mental_health_chatbot/
├── app.py                          # Streamlit application
├── src/
│   ├── rag_system.py              # RAG implementation
│   └── data_processor.py          # Data processing
├── data/
│   ├── veteran_rag_corpus.csv     # Processed corpus
│   ├── female_veterans_clean.csv  # Female veteran data
│   └── male_veterans_clean.csv    # Male veteran data
├── models/
│   └── faiss_index/               # FAISS index files
├── requirements.txt               # Full dependencies
├── requirements-minimal.txt       # Essential dependencies only
├── README.md                      # This file
└── DATA_SOURCES.md               # Complete data documentation
```

---

## Troubleshooting

### Installation Issues

**Problem:** `torch` won't install
```bash
pip install -r requirements-minimal.txt
```

**Problem:** `faiss-cpu` fails
```bash
pip install faiss-cpu --no-cache-dir
```

### Runtime Issues

**Problem:** Index not found
```bash
# Rebuild index (takes 3-5 min)
python src/data_processor.py
streamlit run app.py
```

---

## Team

**Dave & Nipu **

---

**Built with ❤️ for Veterans**
