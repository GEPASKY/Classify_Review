from django.shortcuts import render
from .forms import ReviewForm
from django.http import HttpResponse
import joblib
import os
from django.conf import settings

# Загрузка обученной модели и TF-IDF векторайзера
model = joblib.load('sentiment_model.pkl')  # Укажите путь к вашему файлу модели
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Укажите путь к вашему TF-IDF векторайзеру


def classify_review(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            # Get the review text
            review = form.cleaned_data['review']

            # Transform the review text using the TF-IDF vectorizer
            review_tfidf = vectorizer.transform([review])

            # Get the prediction probability
            proba = model.predict_proba(review_tfidf)[0]  # Returns [probability_negative, probability_positive]

            # Get the rating based on the probability score
            probability_positive = proba[1]

            if probability_positive >= 0.95:  # Strong positive sentiment
                rating = 10
                sentiment = 'Positive'
            elif probability_positive >= 0.75:  # Strong positive sentiment
                rating = 9
                sentiment = 'Positive'
            elif probability_positive >= 0.55:  # Mild positive sentiment
                rating = 7
                sentiment = 'Positive'
            elif probability_positive >= 0.45:  # Neutral sentiment
                rating = 5
                sentiment = 'Neutral'
            elif probability_positive >= 0.25:  # Mild negative sentiment
                rating = 4
                sentiment = 'Negative'
            elif probability_positive >= 0.15:  # Mild negative sentiment
                rating = 3
                sentiment = 'Negative'
            elif probability_positive >= 0.05:  # Mild negative sentiment
                rating = 2
                sentiment = 'Negative'
            else:  # Strong negative sentiment
                rating = 1
                sentiment = 'Negative'

            # Return result to template
            return render(request, 'reviews/classify.html', {
                'form': form,
                'rating': rating,
                'sentiment': sentiment,
                'prediction': True,
            })
    else:
        form = ReviewForm()

    return render(request, 'reviews/classify.html', {'form': form})
