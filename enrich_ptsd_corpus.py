"""
Enrich Veteran Mental Health Corpus with Comprehensive PTSD Content
Adds high-quality, evidence-based information to improve RAG responses
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

# Comprehensive PTSD content for veteran mental health RAG
PTSD_ENRICHMENT_DATA = [
    # ============================================================================
    # PTSD DEFINITIONS & OVERVIEW
    # ============================================================================
    {
        "doc_id": "ptsd_definition_001",
        "chunk_id": "def_001",
        "text": "Post-Traumatic Stress Disorder (PTSD) is a mental health condition that develops after experiencing or witnessing a traumatic event. It is characterized by intrusive memories, avoidance behaviors, negative changes in thinking and mood, and alterations in arousal and reactivity. PTSD affects the brain's stress response systems and can significantly impact daily functioning, relationships, and quality of life.",
        "source": "PTSD Clinical Definition",
        "metadata": {"type": "definition", "category": "overview"},
    },
    {
        "doc_id": "ptsd_definition_002",
        "chunk_id": "def_002",
        "text": "In military veterans, PTSD commonly develops from combat exposure, witnessing death or injury, military sexual trauma, or other service-related traumatic events. The prevalence of PTSD among veterans varies by era: approximately 11-20% of Operations Iraqi Freedom (OIF) and Enduring Freedom (OEF) veterans, 12% of Gulf War veterans, and 15-30% of Vietnam veterans experience PTSD in a given year.",
        "source": "Veteran PTSD Statistics",
        "metadata": {
            "type": "statistics",
            "category": "veterans",
            "population": "military",
        },
    },
    {
        "doc_id": "ptsd_definition_003",
        "chunk_id": "def_003",
        "text": "PTSD is not a sign of weakness. It is a natural psychological response to abnormal, traumatic circumstances. The disorder involves changes in brain structure and function, including alterations in the amygdala (emotion processing), hippocampus (memory), and prefrontal cortex (decision-making). These neurobiological changes explain why PTSD symptoms are not simply a matter of willpower or choice.",
        "source": "PTSD Neurobiology",
        "metadata": {"type": "explanation", "category": "biology"},
    },
    # ============================================================================
    # PTSD SYMPTOMS - DETAILED
    # ============================================================================
    {
        "doc_id": "ptsd_symptoms_001",
        "chunk_id": "sym_001",
        "text": "PTSD symptoms are grouped into four main clusters: (1) Intrusion symptoms include recurrent, involuntary distressing memories, nightmares about the traumatic event, flashbacks where the person feels as if the trauma is recurring, and intense psychological distress or physical reactions to trauma reminders.",
        "source": "PTSD Symptom Criteria DSM-5",
        "metadata": {"type": "symptoms", "category": "intrusion", "cluster": "B"},
    },
    {
        "doc_id": "ptsd_symptoms_002",
        "chunk_id": "sym_002",
        "text": "Avoidance symptoms (Cluster C) include persistent efforts to avoid distressing memories, thoughts, or feelings about the trauma, and avoidance of external reminders such as people, places, conversations, activities, objects, or situations that trigger memories of the traumatic event. Veterans may avoid crowds, driving, loud noises, or specific locations that remind them of deployment.",
        "source": "PTSD Symptom Criteria DSM-5",
        "metadata": {"type": "symptoms", "category": "avoidance", "cluster": "C"},
    },
    {
        "doc_id": "ptsd_symptoms_003",
        "chunk_id": "sym_003",
        "text": "Negative alterations in cognition and mood (Cluster D) include inability to remember important aspects of the trauma, persistent negative beliefs about oneself or the world, distorted blame of self or others, persistent negative emotional state (fear, horror, anger, guilt, shame), diminished interest in activities, feeling detached from others, and persistent inability to experience positive emotions.",
        "source": "PTSD Symptom Criteria DSM-5",
        "metadata": {"type": "symptoms", "category": "mood", "cluster": "D"},
    },
    {
        "doc_id": "ptsd_symptoms_004",
        "chunk_id": "sym_004",
        "text": "Alterations in arousal and reactivity (Cluster E) include irritable behavior and angry outbursts, reckless or self-destructive behavior, hypervigilance (constantly being on guard), exaggerated startle response, problems with concentration, and sleep disturbances. Veterans often describe feeling constantly 'on edge' or unable to relax even in safe environments.",
        "source": "PTSD Symptom Criteria DSM-5",
        "metadata": {"type": "symptoms", "category": "arousal", "cluster": "E"},
    },
    {
        "doc_id": "ptsd_symptoms_005",
        "chunk_id": "sym_005",
        "text": "Common physical symptoms accompanying PTSD include chronic pain, headaches, gastrointestinal problems, cardiovascular issues, and fatigue. Many veterans experience comorbid conditions including depression (50% comorbidity), substance use disorders (particularly alcohol), traumatic brain injury (TBI), and chronic pain conditions.",
        "source": "PTSD Comorbidities",
        "metadata": {"type": "symptoms", "category": "comorbid"},
    },
    # ============================================================================
    # PTSD DIAGNOSIS
    # ============================================================================
    {
        "doc_id": "ptsd_diagnosis_001",
        "chunk_id": "diag_001",
        "text": "PTSD diagnosis requires exposure to actual or threatened death, serious injury, or sexual violence through direct experience, witnessing the event, learning it occurred to a close family member or friend, or experiencing repeated exposure to aversive details of traumatic events (common in first responders). The traumatic event must be followed by symptoms from all four clusters lasting more than one month and causing significant distress or functional impairment.",
        "source": "DSM-5 PTSD Criteria",
        "metadata": {"type": "diagnosis", "category": "criteria"},
    },
    {
        "doc_id": "ptsd_diagnosis_002",
        "chunk_id": "diag_002",
        "text": "Common PTSD screening tools include the PTSD Checklist for DSM-5 (PCL-5), a 20-item self-report measure, and the Primary Care PTSD Screen (PC-PTSD-5), a 5-item screener. The Clinician-Administered PTSD Scale (CAPS-5) is the gold standard diagnostic interview. Veterans Affairs facilities routinely screen for PTSD during clinical encounters.",
        "source": "PTSD Assessment Tools",
        "metadata": {"type": "diagnosis", "category": "screening"},
    },
    # ============================================================================
    # PTSD TREATMENTS - COMPREHENSIVE
    # ============================================================================
    {
        "doc_id": "ptsd_treatment_001",
        "chunk_id": "tx_001",
        "text": "Evidence-based psychotherapies for PTSD include Cognitive Processing Therapy (CPT), Prolonged Exposure (PE) therapy, Eye Movement Desensitization and Reprocessing (EMDR), and Cognitive Behavioral Therapy (CBT). These are considered first-line treatments with strong research support showing significant symptom reduction in 60-80% of patients who complete treatment.",
        "source": "VA/DOD Clinical Practice Guidelines",
        "metadata": {
            "type": "treatment",
            "category": "psychotherapy",
            "evidence": "strong",
        },
    },
    {
        "doc_id": "ptsd_treatment_002",
        "chunk_id": "tx_002",
        "text": "Cognitive Processing Therapy (CPT) is a 12-session structured therapy that helps patients understand and modify unhelpful beliefs related to the trauma. CPT focuses on challenging stuck points - beliefs that keep people from recovering such as self-blame, safety concerns, trust issues, and problems with intimacy and power. CPT can be delivered individually or in groups and has shown efficacy across diverse trauma types and populations.",
        "source": "CPT Clinical Description",
        "metadata": {"type": "treatment", "category": "CPT", "sessions": "12"},
    },
    {
        "doc_id": "ptsd_treatment_003",
        "chunk_id": "tx_003",
        "text": "Prolonged Exposure (PE) therapy involves 8-15 sessions teaching patients to gradually approach trauma-related memories, feelings, and situations that they have been avoiding. PE includes imaginal exposure (repeatedly revisiting the trauma memory) and in vivo exposure (confronting safe situations that have been avoided). PE helps patients process the trauma and learn that trauma-related anxiety decreases over time.",
        "source": "PE Clinical Description",
        "metadata": {"type": "treatment", "category": "PE", "sessions": "8-15"},
    },
    {
        "doc_id": "ptsd_treatment_004",
        "chunk_id": "tx_004",
        "text": "Eye Movement Desensitization and Reprocessing (EMDR) is an 8-phase treatment that uses bilateral stimulation (typically eye movements) while patients process traumatic memories. EMDR helps reprocess traumatic memories so they become less distressing. Treatment typically requires 6-12 sessions and has strong evidence for effectiveness, though the mechanism of action continues to be researched.",
        "source": "EMDR Clinical Description",
        "metadata": {"type": "treatment", "category": "EMDR", "sessions": "6-12"},
    },
    {
        "doc_id": "ptsd_treatment_005",
        "chunk_id": "tx_005",
        "text": "Selective Serotonin Reuptake Inhibitors (SSRIs) are the first-line medication treatment for PTSD. Sertraline (Zoloft) and paroxetine (Paxil) are FDA-approved for PTSD treatment. These medications can help reduce PTSD symptoms including intrusive thoughts, avoidance, negative mood, and hyperarousal. Medications are often most effective when combined with psychotherapy.",
        "source": "PTSD Pharmacotherapy Guidelines",
        "metadata": {"type": "treatment", "category": "medication", "class": "SSRI"},
    },
    {
        "doc_id": "ptsd_treatment_006",
        "chunk_id": "tx_006",
        "text": "Other evidence-based medications for PTSD include SNRIs (venlafaxine), and the antipsychotic risperidone for augmentation when first-line treatments are insufficient. Prazosin is commonly used off-label for PTSD-related nightmares and sleep disturbances. Benzodiazepines are generally not recommended as they may interfere with the processing of traumatic memories and have addiction potential.",
        "source": "PTSD Pharmacotherapy Guidelines",
        "metadata": {"type": "treatment", "category": "medication"},
    },
    {
        "doc_id": "ptsd_treatment_007",
        "chunk_id": "tx_007",
        "text": "Complementary and integrative health approaches for PTSD include yoga, meditation, mindfulness-based stress reduction, acupuncture, and animal-assisted therapy. While evidence is still emerging for these approaches, many veterans find them helpful as adjuncts to evidence-based psychotherapy. The VA offers several complementary therapy programs specifically designed for veterans.",
        "source": "Complementary PTSD Treatments",
        "metadata": {"type": "treatment", "category": "complementary"},
    },
    {
        "doc_id": "ptsd_treatment_008",
        "chunk_id": "tx_008",
        "text": "Treatment for PTSD typically lasts 8-16 weeks for evidence-based psychotherapies. Response to treatment varies, with approximately 30-50% of patients achieving full remission and 60-80% experiencing significant symptom improvement. Early intervention improves outcomes. Treatment dropout rates average 20-30%, often due to avoidance symptoms or logistical barriers.",
        "source": "PTSD Treatment Outcomes",
        "metadata": {"type": "treatment", "category": "outcomes"},
    },
    # ============================================================================
    # VETERAN-SPECIFIC CONTENT
    # ============================================================================
    {
        "doc_id": "veteran_ptsd_001",
        "chunk_id": "vet_001",
        "text": "Combat-related PTSD in veterans often involves moral injury - psychological distress from actions or inactions that violate one's moral code. This may include witnessing or participating in acts that conflict with personal values, betrayal by leadership, or failing to prevent harm. Moral injury can complicate PTSD treatment and requires specialized therapeutic approaches addressing guilt, shame, and meaning-making.",
        "source": "Combat PTSD and Moral Injury",
        "metadata": {
            "type": "veteran-specific",
            "category": "moral injury",
            "population": "combat",
        },
    },
    {
        "doc_id": "veteran_ptsd_002",
        "chunk_id": "vet_002",
        "text": "Military Sexual Trauma (MST) refers to sexual assault or repeated, threatening sexual harassment experienced during military service. Both men and women can experience MST. Approximately 1 in 4 women and 1 in 100 men report MST. Veterans who experienced MST are at higher risk for PTSD, depression, and substance use disorders. VA provides free, confidential treatment for MST-related conditions regardless of discharge status or service connection.",
        "source": "Military Sexual Trauma",
        "metadata": {"type": "veteran-specific", "category": "MST", "gender": "both"},
    },
    {
        "doc_id": "veteran_ptsd_003",
        "chunk_id": "vet_003",
        "text": "Female veterans face unique challenges with PTSD including higher rates of MST, different symptom presentations (more likely to experience depression and anxiety alongside PTSD), and barriers to care such as lack of gender-specific services. Female veterans often experience guilt related to balancing military service with family roles and may face stigma about seeking mental health care in male-dominated veteran communities.",
        "source": "Female Veteran Mental Health",
        "metadata": {
            "type": "veteran-specific",
            "category": "demographics",
            "gender": "female",
        },
    },
    {
        "doc_id": "veteran_ptsd_004",
        "chunk_id": "vet_004",
        "text": "Traumatic Brain Injury (TBI) commonly co-occurs with PTSD in veterans, especially those exposed to blasts. Overlapping symptoms include memory problems, concentration difficulties, irritability, and sleep disturbances. This complicates diagnosis and treatment. Integrated treatment approaches addressing both TBI and PTSD are most effective. The VA provides specialized programs for polytrauma patients with both conditions.",
        "source": "PTSD and TBI Comorbidity",
        "metadata": {"type": "veteran-specific", "category": "TBI", "comorbid": "true"},
    },
    {
        "doc_id": "veteran_ptsd_005",
        "chunk_id": "vet_005",
        "text": "Transition from military to civilian life can exacerbate PTSD symptoms. Veterans may struggle with loss of military identity, difficulty relating to civilians who haven't experienced combat, loss of unit camaraderie, and challenges finding purpose in civilian careers. Reintegration programs and peer support groups specifically designed for veterans can facilitate this transition and reduce PTSD symptom severity.",
        "source": "Veteran Reintegration Challenges",
        "metadata": {"type": "veteran-specific", "category": "transition"},
    },
    # ============================================================================
    # SUPPORT RESOURCES
    # ============================================================================
    {
        "doc_id": "ptsd_resources_001",
        "chunk_id": "res_001",
        "text": "The Veterans Crisis Line provides 24/7 confidential support for veterans in crisis and their families. Call 988 and press 1, text 838255, or chat online at VeteransCrisisLine.net. Veterans do not need to be enrolled in VA benefits or care to use this service. Trained responders connect veterans with mental health professionals and provide immediate crisis intervention.",
        "source": "Veterans Crisis Line",
        "metadata": {"type": "resource", "category": "crisis", "availability": "24/7"},
    },
    {
        "doc_id": "ptsd_resources_002",
        "chunk_id": "res_002",
        "text": "VA provides specialized PTSD treatment programs including PTSD Clinical Teams at all VA medical centers, specialized intensive PTSD programs, residential rehabilitation programs, and telehealth options. Veterans can access mental health services through VA by enrolling in VA health care or through community care programs. Emergency mental health care is available to all veterans regardless of enrollment or discharge status.",
        "source": "VA PTSD Programs",
        "metadata": {"type": "resource", "category": "VA services"},
    },
    {
        "doc_id": "ptsd_resources_003",
        "chunk_id": "res_003",
        "text": "The VA's PTSD Coach mobile app provides education about PTSD, self-assessment tools, symptom tracking, and skills to manage stress and symptoms. The app is free, does not require VA enrollment, and can be used with or without professional treatment. Similar apps include Mindfulness Coach, CBT-i Coach for insomnia, and PE Coach for patients in Prolonged Exposure therapy.",
        "source": "VA Mobile Mental Health Apps",
        "metadata": {"type": "resource", "category": "technology", "cost": "free"},
    },
    {
        "doc_id": "ptsd_resources_004",
        "chunk_id": "res_004",
        "text": "Vet Centers provide free counseling, outreach, and referral services to combat veterans and their families. Services include individual and group counseling, family counseling, bereavement counseling, MST counseling, and community education. Vet Centers are community-based and located separate from VA medical centers to reduce barriers to care. There are over 300 Vet Centers nationwide.",
        "source": "Vet Centers Overview",
        "metadata": {"type": "resource", "category": "vet centers", "cost": "free"},
    },
    # ============================================================================
    # FAMILY & CAREGIVER INFORMATION
    # ============================================================================
    {
        "doc_id": "ptsd_family_001",
        "chunk_id": "fam_001",
        "text": "PTSD affects not only the veteran but also family members and close relationships. Family members often experience secondary traumatization, caregiver burden, and relationship strain. Common challenges include communication difficulties, emotional distance, anger or irritability from the veteran, and changes in family roles. Family therapy and psychoeducation programs can help families understand PTSD and develop coping strategies.",
        "source": "PTSD Impact on Families",
        "metadata": {"type": "family", "category": "impact"},
    },
    {
        "doc_id": "ptsd_family_002",
        "chunk_id": "fam_002",
        "text": "Ways family members can help a veteran with PTSD: Learn about PTSD and its effects; encourage treatment but don't pressure; be patient with recovery which takes time; take care of your own mental health; set boundaries when needed; participate in family therapy if offered; connect with other military families; and remember that PTSD symptoms are not personal attacks. The VA offers education programs specifically for family members.",
        "source": "Supporting Veterans with PTSD",
        "metadata": {"type": "family", "category": "support strategies"},
    },
    {
        "doc_id": "ptsd_family_003",
        "chunk_id": "fam_003",
        "text": "VA offers family support services including the Coaching Into Care program (provides telephone support to family members concerned about a veteran), family therapy as part of PTSD treatment, caregiver support programs, and online resources at www.ptsd.va.gov. Many Vet Centers also offer family counseling and support groups for family members of veterans with PTSD.",
        "source": "VA Family Support Services",
        "metadata": {"type": "family", "category": "VA resources"},
    },
    # ============================================================================
    # PREVENTION & RESILIENCE
    # ============================================================================
    {
        "doc_id": "ptsd_prevention_001",
        "chunk_id": "prev_001",
        "text": "While not all PTSD can be prevented, resilience factors that may reduce risk include strong social support, healthy coping skills, positive pre-trauma mental health, sense of purpose and meaning, physical fitness, and prior experience managing stress. Military resilience programs teach skills including problem-solving, emotional regulation, connection with others, and finding meaning in adversity.",
        "source": "PTSD Prevention and Resilience",
        "metadata": {"type": "prevention", "category": "resilience factors"},
    },
    {
        "doc_id": "ptsd_prevention_002",
        "chunk_id": "prev_002",
        "text": "Early intervention following trauma can prevent development of chronic PTSD. Critical interventions include psychological first aid, screening for early symptoms, ensuring safety and basic needs, connecting with social support, and providing education about normal trauma responses. Avoiding avoidance - gradually returning to normal activities rather than withdrawing - is a key protective factor.",
        "source": "Early PTSD Intervention",
        "metadata": {"type": "prevention", "category": "early intervention"},
    },
    # ============================================================================
    # RECOVERY & PROGNOSIS
    # ============================================================================
    {
        "doc_id": "ptsd_recovery_001",
        "chunk_id": "rec_001",
        "text": "Recovery from PTSD is possible. While some people may experience lingering symptoms, many veterans achieve significant improvement or full recovery with appropriate treatment. Recovery doesn't mean forgetting the trauma, but rather reducing symptom intensity, improving functioning, and reclaiming quality of life. Most veterans who complete evidence-based treatment experience substantial improvement within 3-6 months.",
        "source": "PTSD Recovery Outcomes",
        "metadata": {"type": "recovery", "category": "prognosis"},
    },
    {
        "doc_id": "ptsd_recovery_002",
        "chunk_id": "rec_002",
        "text": "Signs of PTSD recovery include decreased frequency and intensity of intrusive symptoms, improved sleep, better emotional regulation, renewed interest in activities, improved relationships, ability to discuss trauma without overwhelming distress, and increased engagement in meaningful life activities. Recovery is not linear - setbacks are normal and don't mean treatment has failed.",
        "source": "PTSD Recovery Indicators",
        "metadata": {"type": "recovery", "category": "indicators"},
    },
]


def enrich_corpus(
    existing_corpus_path: str = "./data/veteran_rag_corpus.csv",
    output_path: str = "./data/veteran_rag_corpus_enriched.csv",
    backup_original: bool = True,
):
    """
    Enrich existing corpus with comprehensive PTSD content

    Args:
        existing_corpus_path: Path to existing corpus CSV
        output_path: Path for enriched corpus
        backup_original: Whether to backup original file
    """
    print("=" * 80)
    print("VETERAN MENTAL HEALTH CORPUS ENRICHMENT")
    print("=" * 80)

    # Load existing corpus
    existing_path = Path(existing_corpus_path)

    if existing_path.exists():
        print(f"\n✓ Loading existing corpus from: {existing_corpus_path}")
        existing_df = pd.read_csv(existing_corpus_path)
        print(f"  - Existing chunks: {len(existing_df)}")
        print(f"  - Existing sources: {existing_df['source'].nunique()}")

        # Backup original
        if backup_original:
            backup_path = (
                existing_path.parent
                / f"{existing_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            existing_df.to_csv(backup_path, index=False)
            print(f"  - Backup created: {backup_path}")
    else:
        print(f"\n⚠ No existing corpus found at {existing_corpus_path}")
        print("  - Creating new corpus with enrichment data only")
        existing_df = pd.DataFrame()

    # Create enrichment dataframe
    print(f"\n✓ Adding {len(PTSD_ENRICHMENT_DATA)} new PTSD knowledge chunks")
    enrichment_df = pd.DataFrame(PTSD_ENRICHMENT_DATA)

    # Show statistics about enrichment data
    print("\nEnrichment Content Breakdown:")
    print("-" * 80)

    # Count by type
    type_counts = {}
    for item in PTSD_ENRICHMENT_DATA:
        metadata = item.get("metadata", {})
        item_type = metadata.get("type", "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    for item_type, count in sorted(type_counts.items()):
        print(f"  - {item_type.capitalize()}: {count} chunks")

    # Combine dataframes
    if len(existing_df) > 0:
        # Ensure columns match
        enrichment_df = enrichment_df[existing_df.columns]
        combined_df = pd.concat([existing_df, enrichment_df], ignore_index=True)
    else:
        combined_df = enrichment_df

    # Save enriched corpus
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(output_path, index=False)

    print(f"\n✓ Enriched corpus saved to: {output_path}")
    print(f"  - Total chunks: {len(combined_df)}")
    print(f"  - Total sources: {combined_df['source'].nunique()}")
    print(f"  - New chunks added: {len(enrichment_df)}")

    # Show top sources
    print("\nTop 10 Sources by Chunk Count:")
    print("-" * 80)
    top_sources = combined_df["source"].value_counts().head(10)
    for source, count in top_sources.items():
        print(f"  {source}: {count}")

    print("\n" + "=" * 80)
    print("ENRICHMENT COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Update your corpus path in app.py if you saved to a new file")
    print(
        "2. Delete the old FAISS index (./models/faiss_index/) to rebuild with new data"
    )
    print("3. Restart your Streamlit app - it will rebuild the index automatically")
    print("4. Test queries like 'What is PTSD?' to see improved responses")

    return combined_df


def preview_enrichment():
    """Preview the enrichment content without modifying files"""
    print("=" * 80)
    print("PTSD ENRICHMENT CONTENT PREVIEW")
    print("=" * 80)

    categories = {}
    for item in PTSD_ENRICHMENT_DATA:
        source = item["source"]
        if source not in categories:
            categories[source] = []
        categories[source].append(item)

    for source, items in sorted(categories.items()):
        print(f"\n{source} ({len(items)} chunks)")
        print("-" * 80)
        for item in items[:2]:  # Show first 2 from each source
            print(f"\n{item['text'][:200]}...")
        if len(items) > 2:
            print(f"\n... and {len(items) - 2} more chunks")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        # Preview mode
        preview_enrichment()
    else:
        # Enrich mode
        enrich_corpus(
            existing_corpus_path="./data/veteran_rag_corpus.csv",
            output_path="./data/veteran_rag_corpus.csv",  # Overwrite original
            backup_original=True,  # But create backup first
        )
