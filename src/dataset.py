import pandas as pd


# Load dataset
def load_local_dataset():
    dataset = pd.read_csv(
        "C:/Users/Gicano Brothers/Documents/POP Repositories/Sentiment-Analysis/data/raw/IMDB Dataset.csv"
    )
    return dataset
