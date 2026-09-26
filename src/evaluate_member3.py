from pathlib import Path


# ============================================================
# MEMBER 3 — EVALUATION NOTES
# ============================================================

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 70)
    print("MEMBER 3 — EVALUATION / ERROR ANALYSIS REPORT")
    print("=" * 70)

    print("""
This file records the validated evaluation findings for the
current business entity resolution pipeline.

IMPORTANT:
The final matching_results.tsv is a TEST prediction file.
It must NOT be compared directly against train_ground_truth.tsv.

The valid train-side evaluation is currently performed by:

    python src/evaluate_streaming.py

That evaluator uses a sampled training validation set and computes
the competition-style macro F0.5 score.
""")

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print("""
Validation sample:
    2,000 training Source 1 entities

Current matching rule:
    nonblank normalized name frequency <= 10
    OR
    nonblank normalized address frequency <= 4

Macro F0.5:
    0.5028

Exact name/address candidate ceiling:
    3,420 / 6,783 true pairs
    50.42%

Interpretation:
    The current exact name/address candidate generation can expose
    only about half of the true matches in the sampled validation
    pairs. Therefore, candidate generation is a major source of
    recall loss.
""")

    print("=" * 70)
    print("THRESHOLD / CUTOFF COMPARISON")
    print("=" * 70)

    print("""
Configuration                         Macro F0.5
------------------------------------------------
name <= 10 OR address <= 4             0.5028
name <= 10 OR address <= 5             0.5026
name <= 11 OR address <= 4             0.5025
name <= 12 OR address <= 4             0.5024
name <= 14 OR address <= 4             0.5023
name <= 11 OR address <= 5             0.5023
name <= 9  OR address <= 4             0.5023
name <= 10 OR address <= 7             0.5022
name <= 10 OR any address              0.5008
""")

    print("=" * 70)
    print("ERROR PATTERNS OBSERVED")
    print("=" * 70)

    print("""
1. Cross-script / transliteration differences
   Examples include:
   - English <-> Tamil
   - English <-> Hindi
   - English <-> Telugu

2. DBA / AKA aliases
   Example:
   - "Obsidian, LLC"
   - "Korbrixx D.B.A. Obsidian, LLC"

3. Name typos / corrupted text
   Examples include noisy spellings and character substitutions.

4. Word-order changes
   Example:
   - "Orellana Investments LLC"
   - "LLC Orellana Invsmbens"

5. Address-only or address-dominant matches
   Some true matches have weak or unrelated business names but
   highly similar addresses.

6. Address formatting / abbreviation differences
   Examples include:
   - "Fourth Street" vs "4th Street"
   - "Tenth Ave" vs "10th Avenue"
   - shortened locality/state names
   - reordered address components
""")

    print("=" * 70)
    print("MEMBER 3 CONCLUSION")
    print("=" * 70)

    print("""
The sampled validation indicates that simply loosening the existing
exact-name/address frequency cutoffs does not improve macro F0.5.

The dominant observed recall limitations come from:
    - cross-script names
    - aliases / DBA names
    - noisy names
    - address variation
    - matches where address evidence is stronger than name evidence

Because the competition metric is precision-heavy, threshold changes
should be validated on the train split before being applied to the
unlabeled test set.

The current best sampled configuration remains:

    name <= 10 OR address <= 4
    Macro F0.5 = 0.5028
""")


if __name__ == "__main__":
    main()