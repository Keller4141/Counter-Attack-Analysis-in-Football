"""Calculate inter-rater agreement for the human counterattack annotations.

The script reads the two manually annotated DOCX files, converts each rater's
answers into a structured table, calculates majority labels, agreement shares
and Fleiss' kappa, and writes the results to Data/derived/inter_rater_agreement.

For reproducibility, the two DOCX files are stored in Data/annotations/.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parent
DERIVED_DIR = ROOT / "Data" / "derived"
OUTPUT_DIR = DERIVED_DIR / "inter_rater_agreement"
ANNOTATION_DIR = ROOT / "Data" / "annotations"

OLD48_DOCX_PATH = ANNOTATION_DIR / "Kontra vurdering (1).docx"
NEW170_DOCX_PATH = ANNOTATION_DIR / "ANNOTATION_SET.docx"
NEW170_SELECTION_CSV = DERIVED_DIR / "human_annotation_set_50_50_20_20_20_20_current_746_terminal_filtered" / "annotation_selection.csv"

def load_docx_rating_table(path: Path) -> pd.DataFrame:
    # DOCX files are zipped XML files. This extracts the table contents directly
    # without requiring Word to be installed.
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")

    root = ET.fromstring(xml)
    all_rows: list[list[str]] = []
    header = None

    for table_idx, tbl in enumerate(root.findall(".//w:tbl", ns), start=1):
        table_rows: list[list[str]] = []
        for tr in tbl.findall("w:tr", ns):
            row: list[str] = []
            for tc in tr.findall("w:tc", ns):
                texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
                row.append("".join(texts).strip())
            if any(cell for cell in row):
                table_rows.append(row)

        if not table_rows:
            continue

        if table_idx == 1:
            header = table_rows[1]
            data_rows = table_rows[2:]
        else:
            data_rows = table_rows

        all_rows.extend(data_rows)

    if header is None:
        raise ValueError(f"Could not find a table header in {path}")

    df = pd.DataFrame(all_rows, columns=header)
    df = df[df["KLIP"].astype(str).str.fullmatch(r"\d+")].copy()
    df["KLIP"] = df["KLIP"].astype(int)
    df = df.sort_values("KLIP").reset_index(drop=True)
    return df


def extract_docx_text_lines(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fleiss_kappa(binary_counts: pd.DataFrame) -> float:
    """Compute Fleiss' kappa for binary labels.

    Expected columns are n_no_counterattack and n_counterattack.
    """
    if binary_counts.empty:
        raise ValueError("Cannot compute Fleiss' kappa on an empty table.")

    counts = binary_counts.to_numpy(dtype=float)
    n_items, n_categories = counts.shape
    if n_categories != 2:
        raise ValueError("This implementation expects exactly two label categories.")

    ratings_per_item = counts.sum(axis=1)
    if len(set(ratings_per_item)) != 1:
        raise ValueError("Fleiss' kappa requires the same number of ratings per item.")

    n_raters = ratings_per_item[0]
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e_bar = (p_j ** 2).sum()

    if p_e_bar == 1.0:
        return 1.0
    return float((p_bar - p_e_bar) / (1.0 - p_e_bar))


def agreement_distribution(series: pd.Series) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for agreement_votes, count in series.value_counts().sort_index().items():
        distribution[str(int(agreement_votes))] = int(count)
    return distribution


def summarize_votes(item_df: pd.DataFrame, dataset_name: str, id_col: str) -> tuple[pd.DataFrame, dict]:
    excluded_cols = {
        id_col,
        "annotation_error",
        "display_candidate_id",
        "group_name",
        "match_id",
        "start_event_id",
        "team_id",
        "source_label",
    }
    rater_cols = [col for col in item_df.columns if col not in excluded_cols]
    vote_df = item_df[[id_col] + rater_cols].copy()
    for col in rater_cols:
        vote_df[col] = pd.to_numeric(vote_df[col], errors="raise").astype(int)

    # Majority vote becomes the final human label used later in the hybrid model.
    vote_df["positive_votes"] = vote_df[rater_cols].sum(axis=1)
    vote_df["negative_votes"] = len(rater_cols) - vote_df["positive_votes"]
    vote_df["majority_label"] = (vote_df["positive_votes"] >= (len(rater_cols) / 2)).astype(int)
    vote_df["agreement_votes"] = vote_df[["positive_votes", "negative_votes"]].max(axis=1)
    vote_df["agreement_share"] = vote_df["agreement_votes"] / len(rater_cols)
    vote_df["is_unanimous"] = vote_df["agreement_votes"] == len(rater_cols)

    counts_df = vote_df[["negative_votes", "positive_votes"]].rename(
        columns={
            "negative_votes": "n_no_counterattack",
            "positive_votes": "n_counterattack",
        }
    )

    p_i = ((counts_df.to_numpy(dtype=float) ** 2).sum(axis=1) - len(rater_cols)) / (len(rater_cols) * (len(rater_cols) - 1))
    summary = {
        "dataset": dataset_name,
        "n_items": int(len(vote_df)),
        "n_raters": int(len(rater_cols)),
        "rater_names": rater_cols,
        "positive_majority_count": int((vote_df["majority_label"] == 1).sum()),
        "negative_majority_count": int((vote_df["majority_label"] == 0).sum()),
        "unanimous_count": int(vote_df["is_unanimous"].sum()),
        "unanimous_share": float(vote_df["is_unanimous"].mean()),
        "mean_agreement_share": float(vote_df["agreement_share"].mean()),
        "mean_pairwise_agreement": float(p_i.mean()),
        "fleiss_kappa": float(fleiss_kappa(counts_df)),
        "agreement_vote_distribution": agreement_distribution(vote_df["agreement_votes"]),
    }
    return vote_df, summary


def load_old48_votes() -> pd.DataFrame:
    # old48 is the first manually annotated validation set with 48 candidates.
    ratings_df = load_docx_rating_table(OLD48_DOCX_PATH)
    human_cols = [col for col in ratings_df.columns if col not in ["KLIP", "Pipeline"]]
    if len(human_cols) != 7:
        raise ValueError(f"Expected 7 human label columns in old48 file, found {len(human_cols)}: {human_cols}")

    for col in human_cols:
        ratings_df[col] = pd.to_numeric(ratings_df[col], errors="raise").astype(int)

    ratings_df = ratings_df.rename(columns={"KLIP": "clip_id"})
    return ratings_df[["clip_id"] + human_cols].copy()


def load_new170_votes() -> tuple[pd.DataFrame, int]:
    # The 170-case annotation set was exported from Word in a less regular format,
    # so it is parsed from text lines rather than a clean table.
    lines = extract_docx_text_lines(NEW170_DOCX_PATH)
    try:
        christian_idx = lines.index("Christian")
    except ValueError as exc:
        raise RuntimeError(f"Could not locate rater header in {NEW170_DOCX_PATH}") from exc

    rater_names = lines[christian_idx - 6: christian_idx + 1]
    if len(rater_names) != 7:
        rater_names = [f"rater_{idx}" for idx in range(1, 8)]

    start_idx = christian_idx + 1
    rows = []
    error_count = 0
    i = start_idx
    while i < len(lines):
        token = lines[i]
        match = re.match(r"^(\d+)(.*)$", token)
        if not match:
            i += 1
            continue

        candidate_num = int(match.group(1))
        suffix = match.group(2).strip().lower()
        if "fejl" in suffix:
            # "fejl" marks clips that were excluded because of video/tracking issues.
            error_count += 1
            i += 1
            continue

        votes: list[int] = []
        for offset in range(1, 8):
            if i + offset >= len(lines):
                break
            vote = lines[i + offset]
            if vote in {"0", "1"}:
                votes.append(int(vote))
            else:
                break

        if len(votes) != 7:
            raise RuntimeError(f"Could not parse 7 votes for annotation token {token!r} in {NEW170_DOCX_PATH}")

        row = {"display_candidate_num": candidate_num}
        row.update({rater_names[idx]: votes[idx] for idx in range(7)})
        rows.append(row)
        i += 8

    vote_df = pd.DataFrame(rows).sort_values("display_candidate_num").reset_index(drop=True)

    if NEW170_SELECTION_CSV.exists():
        selection_df = pd.read_csv(NEW170_SELECTION_CSV)
        selection_df["display_candidate_num"] = (
            selection_df["display_candidate_id"].str.extract(r"candidate_(\d+)").astype(int)
        )
        vote_df = selection_df[["display_candidate_num", "display_candidate_id", "group_name"]].merge(
            vote_df,
            on="display_candidate_num",
            how="inner",
            validate="one_to_one",
        )

    return vote_df, error_count


def write_summary_files(dataset_key: str, item_df: pd.DataFrame, summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item_path = OUTPUT_DIR / f"{dataset_key}_item_level.csv"
    summary_json_path = OUTPUT_DIR / f"{dataset_key}_summary.json"
    summary_txt_path = OUTPUT_DIR / f"{dataset_key}_summary.txt"

    item_df.to_csv(item_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"Dataset: {summary['dataset']}",
        f"Items: {summary['n_items']}",
        f"Raters: {summary['n_raters']}",
        f"Positive majority items: {summary['positive_majority_count']}",
        f"Negative majority items: {summary['negative_majority_count']}",
        f"Unanimous items: {summary['unanimous_count']} ({summary['unanimous_share']:.3f})",
        f"Mean agreement share: {summary['mean_agreement_share']:.3f}",
        f"Mean pairwise agreement: {summary['mean_pairwise_agreement']:.3f}",
        f"Fleiss' kappa: {summary['fleiss_kappa']:.3f}",
        "Agreement vote distribution:",
    ]
    for votes, count in summary["agreement_vote_distribution"].items():
        lines.append(f"  {votes}/7: {count}")
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    old48_votes = load_old48_votes()
    old48_item_df, old48_summary = summarize_votes(old48_votes, dataset_name="old48_validation_set", id_col="clip_id")

    new170_votes, new170_error_count = load_new170_votes()
    new170_item_df, new170_summary = summarize_votes(new170_votes, dataset_name="new170_annotation_set", id_col="display_candidate_num")
    new170_summary["excluded_error_clips"] = int(new170_error_count)

    write_summary_files("old48", old48_item_df, old48_summary)
    write_summary_files("new170", new170_item_df, new170_summary)

    combined_summary_df = pd.DataFrame([old48_summary, new170_summary])
    combined_summary_path = OUTPUT_DIR / "combined_summary.csv"
    combined_summary_df.to_csv(combined_summary_path, index=False)

    print("Inter-rater agreement summaries")
    for summary in [old48_summary, new170_summary]:
        print(f"\n{summary['dataset']}")
        print(f"  n_items: {summary['n_items']}")
        print(f"  unanimous_share: {summary['unanimous_share']:.3f}")
        print(f"  mean_agreement_share: {summary['mean_agreement_share']:.3f}")
        print(f"  mean_pairwise_agreement: {summary['mean_pairwise_agreement']:.3f}")
        print(f"  fleiss_kappa: {summary['fleiss_kappa']:.3f}")
        if "excluded_error_clips" in summary:
            print(f"  excluded_error_clips: {summary['excluded_error_clips']}")

    print(f"\nSaved: {OUTPUT_DIR / 'old48_item_level.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'old48_summary.json'}")
    print(f"Saved: {OUTPUT_DIR / 'old48_summary.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'new170_item_level.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'new170_summary.json'}")
    print(f"Saved: {OUTPUT_DIR / 'new170_summary.txt'}")
    print(f"Saved: {combined_summary_path}")


if __name__ == "__main__":
    main()
