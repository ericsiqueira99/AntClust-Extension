from afinn import Afinn

# Initialize AFINN
afinn = Afinn()

# Example text
tokens = ["kill", "hurt", "steal", "lie", "hug", "love", "help"]


# Sentiment analysis and encoding
sentiment_scores = [afinn.score(token) for token in tokens]

# Example sentiment scores for tokens
sentiment_scores_dict = dict(zip(tokens, sentiment_scores))
print(sentiment_scores_dict)
