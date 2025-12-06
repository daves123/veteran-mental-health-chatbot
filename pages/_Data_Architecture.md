<!-- # BRFSS RAG Chatbot: Data Architecture Decisions

## Executive Summary

Our RAG system maintains **separate indices for BRFSS 2023 and 2024 data** rather than combining them into a single dataset. This decision is driven by the cross-sectional nature of BRFSS and the need for temporal accuracy in responses.

## 1. Why We Did NOT Combine Years

### A. Temporal Query Accuracy
Users frequently ask year-specific questions:
- "What was the smoking rate in California in 2024?"
- "Show me the latest diabetes statistics"
- "How did obesity rates change from 2023 to 2024?"

Keeping years separate with metadata tags allows our RAG system to:
- Retrieve the exact year requested
- Differentiate between "latest" (2024) vs. historical (2023)
- Prevent contamination where 2023 data might answer "current" queries

### B. Trend Analysis Capability
Separate indexing enables:
- Year-over-year comparisons
- Trend identification
- Context about improving/worsening conditions

### C. Data Provenance
Users need to know:
- Which year their data comes from
- Whether they're seeing current information
- How reliable/recent the answer is

### D. Statistical Correctness
- Each year = different random sample
- Inappropriate to merge independent samples
- Respects cross-sectional methodology

### E. Scalability
- Easy to add 2025+ data
- Maintains historical archive
- Supports multi-year trend queries

## 2. Why BRFSS Records Are Unique Each Year

### Cross-Sectional Survey Design

BRFSS is a **repeated cross-sectional survey**, NOT longitudinal:

**Key Facts:**
- Uses random-digit dialing EACH YEAR for NEW participants
- Does NOT follow the same people over time
- Completely anonymous (no phone numbers, no IDs stored)
- Different people surveyed each year
- Cannot link individuals across years (by design)

**From CDC Documentation:**
> "BRFSS is a cross-sectional telephone survey"

### Statistical Implications

**What You CAN Do:**
✓ Compare population-level prevalence: "Smoking decreased from 15.3% (2023) to 14.8% (2024)"
✓ Assess trends: "PTSD prevalence has increased over time"
✓ Examine relationships: "Association between exercise and heart disease"

**What You CANNOT Do:**
✗ Track individuals: "How did John's health change?"
✗ Calculate transitions: "What % of smokers quit between years?"
✗ Use paired statistical tests across years

### Why This Design?

**Advantages:**
- Large sample sizes (400,000+ annually)
- Cost-effective vs. longitudinal tracking
- Quick annual results
- Strong privacy protection
- Representative population snapshots

**Trade-offs:**
- Cannot assess individual-level causality as well
- Cannot track behavior change in individuals
- Misses within-person variation

## 3. Implementation in Our RAG System

### Metadata Structure
```python
document_metadata = {
    "year": 2024,
    "state": "California",
    "topic": "mental_health",
    "gender": "male",
    "source": "BRFSS_2024"
}
```

### Retrieval Strategy
```python
# Year-specific query
results = vector_store.similarity_search(
    query=user_question,
    filter={"year": 2024}
)

# Multi-year trend query
results = vector_store.similarity_search(
    query=user_question,
    filter={"year": {"$gte": 2023}}
)
```

### When We DO Aggregate
The chatbot combines information across years only when:
1. User explicitly asks for trends/comparisons
2. Answering methodological questions about BRFSS
3. Providing historical context
4. Computing multi-year statistics (with proper methods)

## 4. Impact on Chatbot Responses

### Correct Responses:
- "In 2024, 28.3% of Texas adults reported obesity"
- "Smoking rates in California decreased from 11.2% in 2023 to 10.8% in 2024"
- "The data shows population-level trends across independent samples"

### Incorrect Responses:
- ❌ "People who smoked in 2023 had X% quit rate in 2024"
- ❌ "The same respondents showed improvement"
- ❌ Language implying following same people over time

## 5. References

- CDC BRFSS Overview: https://www.cdc.gov/brfss/
- BRFSS described as "cross-sectional telephone survey"
- Random-digit dialing used annually
- No personally identifiable information collected
- Each year = independent probability sample

## 6. Future Enhancements

- Multi-year trend visualization
- Smart year detection ("latest" → 2024)
- Automatic year-comparison mode
- Data recency indicators
- Built-in methodology explanations

---

**Authors:** Dave Singh & Nipu Quayum  
**Project:** Data Science Masters - Veteran Mental Health RAG Chatbot  
**Date:** December 2024 -->