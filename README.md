# Recommender GUI using: BRP (Bayesian Personalized Ranking)

## Requirements
- Python 3.12.0

## How to use
1. `python -m venv bpr-gui`
1. `bpr-gui\Scripts\activate`
1. `pip install -r requirements.txt`
1. `python app.py`
1. Have an excel ready with the following columns: `user_id`, `item_id`, `rating`
1. Load input excel sheet
1. Train BPR model: This will produce a file with the results on test set that is 
extracted from the dataset.
1. Export All Recommendations: will export the recommendations to excel file
