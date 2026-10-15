"""
Cross-country feature harmonization for Phase 12: maps Pakistan (PERI_DEP)
and Arab5 (Qasrawi et al.) onto a common 5-feature SDoH schema, enabling
the Phase 13 cross-national external validation.

IMPORTANT CONTEXT: Arab5's numeric-looking columns (WomenAge, MariageAge,
NumberofPregnancy, NumberofAbortions, EducationLevel, Work, FamilyIncome1,
SleepingAfter) are coded ORDINAL CATEGORIES, not continuous measurements --
confirmed against the dataset's own Variables_Descriptions.xlsx value-label
sheet. This module encodes those exact mappings.

DOCUMENTED LIMITATION: financial_strain is an imperfect construct match --
Pakistan's "Sufficient Money for Basic Needs" measures absolute financial
adequacy, while Arab5's "FamilyIncome1" measures pandemic-era income TREND
(decreased vs. increased/same), not absolute sufficiency. Treated as
broadly analogous (both indicate financial precarity vs. relative
stability) but should be flagged as a proxy, not a clean equivalence, in
the manuscript's Methods/Limitations.
"""
import numpy as np
import pandas as pd


def harmonize_pakistan(pk_df):
    """
    Map Pakistan (PERI_DEP) columns onto the common harmonized schema.
    Expects pk_df to already have whitespace-cleaned categorical columns
    (see the Phase 3 cleaning fix) and a y_depression column.
    """
    h = pd.DataFrame(index=pk_df.index)

    age_group = pd.Series(np.where(pk_df["Age"] < 35, "under_35", "35_plus"), index=pk_df.index)
    age_group[pk_df["Age"] < 15] = np.nan  # exclude implausible age values (e.g. Age=3, a data-entry error)
    h["age_group"] = age_group

    edu_map = {
        "Uneducated": "secondary_or_less", "Primary": "secondary_or_less",
        "Middle": "secondary_or_less", "Matric": "secondary_or_less",
        "Intermediate": "above_secondary", "Graduation": "above_secondary",
    }
    h["education_group"] = pk_df["Female Education"].str.strip().map(edu_map)

    work_map = {"Housewife": "not_employed", "Working Lady": "employed"}
    h["employed"] = pk_df["Working Status"].str.strip().map(work_map)

    fin_map = {"No": "strain", "Yes": "no_strain"}
    h["financial_strain"] = pk_df["Sufficient Money for Basic Needs"].str.strip().map(fin_map)

    grav_map = {"Primigravida": "first_pregnancy", "Multigravida": "not_first",
                "Multigravida+D1191": "not_first"}  # pre-Phase-3-cleaning-fix artifact, mapped defensively
    h["first_pregnancy"] = pk_df["Gravida"].map(grav_map)

    h["y_depression"] = pk_df["y_depression"]
    h["source_country"] = "Pakistan"
    return h


def harmonize_arab5(ar_df):
    """
    Map Arab5 (Qasrawi et al.) columns onto the common harmonized schema,
    using the exact numeric-code-to-label mapping from the dataset's own
    Variables_Descriptions.xlsx.
    """
    h = pd.DataFrame(index=ar_df.index)

    h["age_group"] = ar_df["WomenAge"].map({1: "under_35", 2: "35_plus"})
    h["education_group"] = ar_df["EducationLevel"].map({1: "secondary_or_less", 2: "above_secondary"})
    h["employed"] = ar_df["Work"].map({1: "employed", 2: "not_employed"})
    h["financial_strain"] = ar_df["FamilyIncome1"].map({1: "strain", 2: "no_strain"})
    h["first_pregnancy"] = ar_df["NumberofPregnancy"].map({
        1: "first_pregnancy", 2: "not_first", 3: "not_first", 4: "not_first"})

    h["y_depression"] = ar_df["y_depression"]
    h["source_country"] = ar_df["part1_country"].map({
        1: "Jordan", 2: "Palestine", 4: "Lebanon", 6: "Saudi Arabia", 8: "Bahrain"})
    if "dep_items_complete" in ar_df.columns:
        h["dep_items_complete"] = ar_df["dep_items_complete"]
    return h


HARMONIZED_FEATURE_COLS = ["age_group", "education_group", "employed", "financial_strain", "first_pregnancy"]
