Atelier – Clothing Review Recommendation System

See README.md for full portfolio documentation, evaluation metrics, and setup instructions.

Originally developed as Assignment 3 Milestone II – NLP Web-based Data Application
(Advanced Programming for Data Science)

## Project Structure 
app.py                  Main Flask application
assignment3_II.csv      Clothing reviews dataset (Milestone II version)
stopwords_en.txt        English stopwords used in preprocessing pipeline
requirements.txt        Python dependencies 
templates/
    base.html           Shared layout and navigation
    index.html          Homepage paginated item listings
    category.html       Category browsing by Department Name
    item.html           Item detail page with all reviews
    search.html         Search results page
    review_form.html    Create new review form
    review_predict.html ML prediction display with override option
    review_detail.html  Confirmed review
    404.html            404 error page
static/css/style.css    Stylesheet

## How to Run
1. Navigate into the app folder:
      cd app

2. Create and activate a virtual environment:
      python -m venv venv
      .\venv\Scripts\activate            # Windows

3. Install dependencies:
      pip install -r requirements.txt

4. Run the application:
      python app.py

5. Open a browser and navigate to:
      http://127.0.0.1:5000


## ML Model Details 
Language model   : Bag-of-Words (CountVectorizer, min_df=2)
Input            : Title + Review Text (concatenated, preprocessed)
Classifier       : Logistic Regression (max_iter=1000, random_state=42)
Training source  : assignment3_II.csv (19,662 reviews)
Accuracy         : ~89% 
Preprocessing    : tokenize with r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
                   lowercase, remove single-char tokens, remove stopwords

## Search Algorithm 
Keyword search applies Porter stemming (NLTK PorterStemmer) to both the
query and the item's title/description/category text. Matches are
ranked by the number of stemmed token hits per item.

## New Review Storage
New reviews are saved to new_reviews.csv and loaded when the app starts.
Each review is assigned a UUID-based ID and is accessible via the URL
/review/<review_id>. Reviews appear immediately on the item's detail page
and remain available after restarting the server.


## Notes 

- No customer login is required.
- The model is retrained at startup from the CSV, no serialised model file is needed. Startup takes ~5-10 seconds.


## References
- Flask documentation: https://flask.palletsprojects.com/
- scikit-learn documentation: https://scikit-learn.org/
- NLTK PorterStemmer: https://www.nltk.org/
- Dataset source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

