from django.shortcuts import render
from .forms import ReviewForm
from django.http import HttpResponse
import joblib
import os
from django.conf import settings

#from ..movie_reviews import settings

# Загрузка обученной модели и TF-IDF векторайзера
model = joblib.load('sentiment_model.pkl')  # Укажите путь к вашему файлу модели
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Укажите путь к вашему TF-IDF векторайзеру


def classify_review(request):
    template_path = os.path.join(settings.BASE_DIR, 'reviews/templates/reviews/classify.html')
    print("Template path:", template_path)

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            # Получаем текст отзыва из формы
            review = form.cleaned_data['review']

            # Преобразуем отзыв с помощью TF-IDF векторайзера
            review_tfidf = vectorizer.transform([review])

            # Прогнозируем с помощью загруженной модели
            prediction = model.predict(review_tfidf)[0]

            # Присваиваем рейтинг (например, от 1 до 10) в зависимости от оценки
            if prediction == 1:
                rating = 8  # Пример рейтинга для положительного отзыва
                sentiment = 'Positive'
            else:
                rating = 3  # Пример рейтинга для отрицательного отзыва
                sentiment = 'Negative'

            # Возвращаем результат в шаблон
            return render(request, 'reviews/classify.html', {
                'form': form,
                'prediction': prediction,
                'rating': rating,
                'sentiment': sentiment,
            })
    else:
        form = ReviewForm()

    return render(request, 'reviews/classify.html', {'form': form})
