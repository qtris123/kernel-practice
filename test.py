import pandas as pd
from functools import reduce

def summarize_school_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
      df with columns:
        - school_id
        - state_code
        - subjects   (space-separated, lowercase)
    Output:
      DataFrame indexed by cleaned state_code, with columns:
        ['English', 'Maths', 'Physics', 'Chemistry']
      Values are counts of schools in that state offering the subject.
    """

    # 1) Basic sanity: ensure expected columns exist
    required = {"school_id", "state_code", "subjects"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()

    # 2) Clean state_code to alphanumeric only; drop rows that become empty
    work["state_code"] = (
        work["state_code"]
        .astype(str)
        .str.replace(r"[^0-9A-Za-z]", "", regex=True)
    )
    work = work[work["state_code"].str.len() > 0]

    # 3) Normalize and parse subjects into a set per row (unique subjects per school)
    #    - tolerate NaN
    #    - collapse multiple spaces
    #    - drop empty tokens
    subj = (
        work["subjects"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    work["subjects_set"] = subj.apply(lambda s: set(t for t in s.split(" ") if t))

    # 4) (Robust) Deduplicate possible multiple rows per school by unioning subjects.
    #    Keep subjects union for (school_id, state_code) pair.
    #    If a school appears under two different state_codes, treat them separately.
    work = (
        work.groupby(["school_id", "state_code"], as_index=False)
            .agg(subjects_set=("subjects_set", lambda s: set().union(*s)))
    )

    # 5) Drop schools offering fewer than 3 UNIQUE subjects
    work = work[work["subjects_set"].apply(lambda st: len(st) >= 3)].copy()

    # 6) Flag membership for target subjects (count each school at most once)
    def has_english(st): return "english" in st
    def has_maths(st):   return ("maths" in st) or ("math" in st)  # accept both
    def has_physics(st): return "physics" in st
    def has_chem(st):    return "chemistry" in st

    work["English"]   = work["subjects_set"].apply(lambda st: int(has_english(st)))
    work["Maths"]     = work["subjects_set"].apply(lambda st: int(has_maths(st)))
    work["Physics"]   = work["subjects_set"].apply(lambda st: int(has_physics(st)))
    work["Chemistry"] = work["subjects_set"].apply(lambda st: int(has_chem(st)))

    # 7) Group by state and sum the flags
    out = (
        work.groupby("state_code", as_index=True)[["English", "Maths", "Physics", "Chemistry"]]
            .sum()
            .sort_index()
    )
    out = 

    # (Optional) Ensure integer dtype
    return out.astype("int64")

data = {
    "school_id":  [1, 2, 3, 4, 5, 5],
    "state_code": ["CA-1", "CA#1", "NY!", "TX", "TX", "TX"],
    "subjects": [
        "english maths physics",      # 3 subjects -> counts EN, MA, PH for CA1
        "english chemistry",          # 2 subjects -> dropped
        "math physics chemistry",     # 3 subjects -> counts MA, PH, CH for NY
        "english  english  maths",    # 2 unique subjects -> dropped
        "english maths chemistry",    # row 1 of school 5
        "physics"                     # row 2 of school 5, total union -> 4 unique -> counted for TX all four except maybe PH? yes includes PH
    ],
}
df = pd.DataFrame(data)

print(summarize_school_subjects(df))
