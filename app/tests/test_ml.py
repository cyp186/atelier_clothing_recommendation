import pandas as pd
import pytest

from ml import build_pipeline, predict_recommendation, train_pipeline
from nlp import preprocess, search_item_score, stem_query, stem_word


class TestPreprocess:
    def test_lowercases_and_removes_stopwords(self):
        result = preprocess("The Dress is Beautiful and Perfect")
        assert "the" not in result.split()
        assert "dress" in result.split()
        assert "beautiful" in result.split()

    def test_handles_non_string(self):
        assert preprocess(None) == ""
        assert preprocess(42) == ""

    def test_keeps_hyphenated_tokens(self):
        result = preprocess("well-made top")
        assert "well-made" in result.split()


class TestStemming:
    def test_stem_query_returns_stems(self):
        stems = stem_query("running dresses")
        assert stem_word("running") in stems
        assert stem_word("dresses") in stems


class TestSearch:
    def test_search_item_score_counts_matches(self):
        row = {
            "clothes_title": "Silk Summer Dress",
            "clothes_desc": "A lightweight dress for warm weather",
            "class_name": "Dresses",
            "department": "General",
        }
        score = search_item_score(row, stem_query("summer dress"))
        assert score >= 2

    def test_search_item_score_zero_for_no_match(self):
        row = {
            "clothes_title": "Winter Coat",
            "clothes_desc": "Warm outerwear",
            "class_name": "Jackets",
            "department": "General",
        }
        assert search_item_score(row, stem_query("swimsuit")) == 0


class TestModel:
    @pytest.fixture
    def tiny_df(self):
        return pd.DataFrame({
            "Title": [
                "Love it", "Terrible fit", "Great quality", "Runs small",
                "Perfect dress", "Awful fabric", "Nice color", "Would buy again",
            ],
            "Review Text": [
                "Absolutely recommend this item to everyone",
                "Do not recommend this at all very disappointed",
                "Highly recommend great purchase",
                "Not recommended poor stitching",
                "Recommend buying this beautiful dress",
                "Would not recommend waste of money",
                "Recommend this for summer events",
                "Definitely recommend excellent value",
            ],
            "Recommended IND": [1, 0, 1, 0, 1, 0, 1, 1],
        })

    def test_train_and_predict(self, tiny_df):
        pipeline = train_pipeline(tiny_df)
        label, confidence = predict_recommendation(
            pipeline,
            "Great item",
            "I highly recommend this product",
        )
        assert label in (0, 1)
        assert 0 <= confidence <= 100

    def test_preprocess_consistency_between_train_and_predict(self, tiny_df):
        pipeline = build_pipeline("count", "logistic")
        from ml import prepare_training_frame

        frame = prepare_training_frame(tiny_df)
        pipeline.fit(frame["processed"], frame["Recommended IND"].values)

        title = "Lovely"
        text = "I recommend this dress"
        label1, _ = predict_recommendation(pipeline, title, text)

        combined = preprocess(title + " " + text)
        label2 = int(pipeline.predict([combined])[0])
        assert label1 == label2
