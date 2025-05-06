# predictor_api/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register our viewsets with it.
router = DefaultRouter()
router.register(r"teams", views.TeamViewSet, basename="team")
router.register(r"venues", views.VenueViewSet, basename="venue")
router.register(r"matches", views.MatchViewSet, basename="match")
router.register(r"players", views.PlayerViewSet, basename="player")

# The API URLs are now determined automatically by the router.
urlpatterns = [
    # Include the router-generated URLs for Teams, Venues, Matches
    path("", include(router.urls)),
    # Add the specific path for your PredictionView
    # Use 'predict/' or your preferred endpoint name
    path("predict/", views.PredictionView.as_view(), name="predict_match"),
    path("predict_score/", views.ScorePredictionView.as_view(), name="predict_score"),
    path("llm-query/", views.LLMQueryView.as_view(), name="llm_query"),
]
