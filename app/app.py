"""
Atelier – Clothing review web app with ML-powered recommendation prediction.
"""

import re
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from ml import get_or_train_model, predict_recommendation
from nlp import search_item_score, stem_query

app = Flask(__name__)
DATA_DIR = Path(__file__).parent
NEW_REVIEWS_FILE = DATA_DIR / "new_reviews.csv"

RAW_DF = pd.read_csv(DATA_DIR / "assignment3_II.csv")

NEW_REVIEWS_COLS = [
    "review_id", "Clothing ID", "Age", "Title", "Review Text", "Rating",
    "Recommended IND", "Positive Feedback Count", "Division Name",
    "Department Name", "Class Name", "Clothes Title", "Clothes Description",
]


def _normalize_review_types(review: dict) -> dict:
    review["Clothing ID"] = int(review["Clothing ID"])
    review["Age"] = int(review["Age"]) if str(review["Age"]).isdigit() else 0
    review["Rating"] = int(review["Rating"])
    review["Recommended IND"] = int(review["Recommended IND"])
    review["Positive Feedback Count"] = int(review.get("Positive Feedback Count", 0) or 0)
    return review


def load_reviews_from_file() -> list[dict]:
    if not NEW_REVIEWS_FILE.exists() or NEW_REVIEWS_FILE.stat().st_size == 0:
        return []

    try:
        saved = pd.read_csv(NEW_REVIEWS_FILE, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        print("Warning: new_reviews.csv is unreadable; starting with no saved reviews.")
        return []

    if saved.empty or not set(NEW_REVIEWS_COLS).issubset(saved.columns):
        print("Warning: new_reviews.csv has an invalid format; starting with no saved reviews.")
        return []

    reviews = [_normalize_review_types(dict(r)) for r in saved.to_dict("records")]
    print(f"Loaded {len(reviews)} saved review(s) from file.")
    return reviews


NEW_REVIEWS: list[dict] = load_reviews_from_file()


def save_reviews_to_file() -> None:
    if not NEW_REVIEWS:
        return
    pd.DataFrame(NEW_REVIEWS, columns=NEW_REVIEWS_COLS).to_csv(
        NEW_REVIEWS_FILE,
        index=False,
    )


def get_reviews_df() -> pd.DataFrame:
    if not NEW_REVIEWS:
        return RAW_DF.copy()
    new_df = pd.DataFrame(NEW_REVIEWS).drop(columns=["review_id"], errors="ignore")
    return pd.concat([RAW_DF, new_df], ignore_index=True)


def get_items_df() -> pd.DataFrame:
    df = get_reviews_df()
    items = (
        df.groupby("Clothes Title")
        .agg(
            clothing_id=("Clothing ID", "first"),
            clothes_desc=("Clothes Description", "first"),
            class_name=("Class Name", "first"),
            department=("Department Name", "first"),
            division=("Division Name", "first"),
            avg_rating=("Rating", "mean"),
            review_count=("Review Text", "count"),
            recommended_pct=("Recommended IND", "mean"),
        )
        .reset_index()
        .rename(columns={"Clothes Title": "clothes_title", "clothing_id": "Clothing ID"})
    )
    items["avg_rating"] = items["avg_rating"].round(1)
    items["recommended_pct"] = (items["recommended_pct"] * 100).round(0).astype(int)
    return items


print("Loading recommendation model...")
MODEL = get_or_train_model(RAW_DF)
print("Model ready.")


@app.route("/")
def index():
    items = get_items_df()
    page = request.args.get("page", 1, type=int)
    per_page = 12
    total = len(items)
    start = (page - 1) * per_page
    departments = sorted(RAW_DF["Department Name"].unique())
    return render_template(
        "index.html",
        items=items.iloc[start:start + per_page].to_dict("records"),
        page=page,
        total_pages=(total + per_page - 1) // per_page,
        total_items=total,
        departments=departments,
    )


@app.route("/category/<department>")
def category(department):
    items = get_items_df()
    filtered = items[items["department"] == department].to_dict("records")
    classes = sorted(RAW_DF[RAW_DF["Department Name"] == department]["Class Name"].unique())
    departments = sorted(RAW_DF["Department Name"].unique())
    return render_template(
        "category.html",
        items=filtered,
        department=department,
        classes=classes,
        departments=departments,
        count=len(filtered),
    )


@app.route("/item/<int:item_id>")
def item_detail(item_id):
    df = get_reviews_df()
    seed = df[df["Clothing ID"] == item_id]
    if seed.empty:
        return render_template("404.html"), 404

    clothes_title = seed["Clothes Title"].iloc[0]
    all_rows = df[df["Clothes Title"] == clothes_title]

    item_info = {
        "id": item_id,
        "title": clothes_title,
        "description": all_rows["Clothes Description"].iloc[0],
        "class_name": all_rows["Class Name"].iloc[0],
        "department": all_rows["Department Name"].iloc[0],
        "division": all_rows["Division Name"].iloc[0],
        "avg_rating": round(all_rows["Rating"].mean(), 1),
        "review_count": len(all_rows),
        "recommended_pct": int(all_rows["Recommended IND"].mean() * 100),
    }
    dist = all_rows["Rating"].value_counts().sort_index(ascending=False)
    rating_dist = {int(k): int(v) for k, v in dist.items()}
    reviews = all_rows[["Title", "Review Text", "Rating", "Recommended IND", "Age"]].to_dict("records")
    departments = sorted(RAW_DF["Department Name"].unique())
    return render_template(
        "item.html",
        item=item_info,
        reviews=reviews,
        rating_dist=rating_dist,
        total_reviews=len(all_rows),
        departments=departments,
    )


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    departments = sorted(RAW_DF["Department Name"].unique())
    if not query:
        return render_template("search.html", items=[], query="", count=0, departments=departments)

    stemmed = stem_query(query)
    items = get_items_df()
    items["score"] = items.apply(lambda row: search_item_score(row, stemmed), axis=1)
    matched = items[items["score"] > 0].sort_values("score", ascending=False)
    return render_template(
        "search.html",
        items=matched.to_dict("records"),
        query=query,
        count=len(matched),
        departments=departments,
    )


@app.route("/review/new/<int:item_id>", methods=["GET", "POST"])
def new_review(item_id):
    df = get_reviews_df()
    item_rows = df[df["Clothing ID"] == item_id]
    if item_rows.empty:
        return render_template("404.html"), 404

    item_info = {
        "id": item_id,
        "title": item_rows["Clothes Title"].iloc[0],
        "class_name": item_rows["Class Name"].iloc[0],
    }
    departments = sorted(RAW_DF["Department Name"].unique())

    if request.method == "POST":
        review_title = request.form.get("title", "").strip()
        review_text = request.form.get("review_text", "").strip()
        rating = int(request.form.get("rating", 3))
        age = request.form.get("age", "").strip()
        pred_label, confidence = predict_recommendation(MODEL, review_title, review_text)
        return render_template(
            "review_predict.html",
            item=item_info,
            review_title=review_title,
            review_text=review_text,
            rating=rating,
            age=age,
            pred_label=str(pred_label),
            confidence=confidence,
            departments=departments,
        )

    return render_template("review_form.html", item=item_info, departments=departments)


@app.route("/review/confirm", methods=["POST"])
def confirm_review():
    item_id = int(request.form.get("item_id"))
    review_title = request.form.get("title", "")
    review_text = request.form.get("review_text", "")
    rating = int(request.form.get("rating", 3))
    age_raw = request.form.get("age", "0")
    age = int(age_raw) if age_raw.isdigit() else 0
    final_label = int(request.form.get("final_label", 0))

    item_row = RAW_DF[RAW_DF["Clothing ID"] == item_id].iloc[0]
    review_id = str(uuid.uuid4())[:8]

    new_review = {
        "review_id": review_id,
        "Clothing ID": item_id,
        "Age": age,
        "Title": review_title,
        "Review Text": review_text,
        "Rating": rating,
        "Recommended IND": final_label,
        "Positive Feedback Count": 0,
        "Division Name": item_row["Division Name"],
        "Department Name": item_row["Department Name"],
        "Class Name": item_row["Class Name"],
        "Clothes Title": item_row["Clothes Title"],
        "Clothes Description": item_row["Clothes Description"],
    }

    NEW_REVIEWS.append(new_review)
    save_reviews_to_file()
    return redirect(url_for("review_detail", review_id=review_id))


@app.route("/review/<review_id>")
def review_detail(review_id):
    review = next((r for r in NEW_REVIEWS if r["review_id"] == review_id), None)
    if review is None:
        return render_template("404.html"), 404
    departments = sorted(RAW_DF["Department Name"].unique())
    return render_template("review_detail.html", review=review, departments=departments)


if __name__ == "__main__":
    app.run(debug=True)
