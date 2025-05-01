from django.urls import path
from . import views  # Import the views module

app_name = "predictor_api"

# Define type hint for urlpatterns
urlpatterns: list = [
    # Map the URL 'predict/' to our PredictionView
    # .as_view() is used for class-based views like APIView
    path("predict/", views.PredictionView.as_view(), name="predict"),
]
