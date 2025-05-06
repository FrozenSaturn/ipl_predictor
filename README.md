# IPL Cricket Prediction System & Analytics Platform

## 1. Project Overview

This project is a comprehensive IPL cricket prediction system designed to forecast match outcomes and scores, integrating machine learning with large language model (LLM) reasoning for explainable AI. It features a robust backend built with both Django and FastAPI, a detailed database schema, and an interactive React-based frontend for user engagement and data visualization.

This system addresses the technical assessment requirements by delivering a functional AI-driven prediction engine, a well-structured API, and an intuitive user interface.

## 2. Core Features

* **Match Outcome Prediction:** Predicts the winner of IPL matches using a trained machine learning model (Random Forest Classifier).
* **First Innings Score Prediction:** Forecasts the score of the first innings using a regression model (XGBRegressor).
* **Explainable AI (XAI):** Integrates with a local LLM (Ollama - `phi3:instruct`) to provide natural language explanations for match winner predictions, detailing key influencing factors.
* **Dual Backend Implementation:** Prediction logic is served via identical API endpoints deployed on both Django (using Django REST Framework) and FastAPI, demonstrating flexible backend architecture.
* **Comprehensive Django Backend:**
    * Relational database schema (PostgreSQL) for storing detailed match, team, player, and granular player-per-match performance statistics.
    * Automated data population from provided CSVs via custom Django management commands.
    * Secure, token-authenticated RESTful APIs for data retrieval and predictions.
    * Automatic OpenAPI 3.0 documentation via `drf-spectacular` (Swagger UI & ReDoc).
* **Interactive React Frontend (Bonus):**
    * Modern, responsive "Bento Box" dark-themed UI.
    * Chatbot interface for initiating predictions and interacting with the system.
    * **Unique:** Contextual LLM follow-up Q&A, allowing users to probe deeper into prediction reasoning.
    * **Unique:** Animated gauge visualization for displaying prediction confidence.
    * **Unique:** Player performance trend visualizations (batting/bowling stats for last 5 matches) integrated directly into the chat and a dedicated UI panel.
    * Head-to-head team statistics accessible via chat commands.

## 3. Technical Implementation

### 3.1. AI Model Development (Task 1)

* **Objective:** Develop an AI model to predict match outcomes and scores, with LLM-based reasoning.
* **Data Sources Used:** `Match_Info.csv` and `Ball_By_Ball_Match_Data.csv`.
* **Feature Engineering:**
    * A `build_features.py` script pre-calculates and consolidates features chronologically to avoid redundant computations.
    * Features include: basic categorical data (teams, venue, toss), historical team win percentages (overall and H2H), previous match statistics (score, wickets fetched from the database), and team-level aggregated recent player form (batting SR, bowling Economy from database PlayerMatchPerformance).
* **Modeling:**
    * Scikit-learn Pipelines are used for chaining preprocessing (OneHotEncoding, StandardScaler) and modeling.
    * **Match Winner Prediction:** A tuned Random ForestClassifier (hyperparameters optimized via `RandomizedSearchCV` focusing on F1 Macro score). Class imbalance was addressed using `class_weight='balanced'`.
    * **Score Prediction:** XGBRegressor.
    * Models and encoders are saved as `.joblib` files and loaded by a central `predictor.py` module.
* **Prediction Logic (`src/ipl_predictor/predictor.py`):**
    * Initializes Django environment for database access.
    * Lazy-loads trained model pipelines.
    * Implements asynchronous database lookups (`sync_to_async`) to fetch pre-calculated historical and player form features from Django models at prediction time.
* **LLM Integration:**
    * Interacts with a locally hosted Ollama model (`phi3:instruct`).
    * Dynamically generates prompts for winner prediction explanations, including match context, key statistics, and the model's prediction. Prompt engineering was iterated upon for improved relevance.
* **Dual Framework Service:** The core prediction logic in `predictor.py` is utilized by both Django and FastAPI API endpoints to serve predictions.

### 3.2. Backend Development (Task 2 - Django)

* **Objective:** Develop a robust Django backend with a well-defined API and database.
* **Database Design (`predictor_api/models.py`):**
    * Models: `Venue`, `Team`, `Player`, `Match`.
    * Specialized `PlayerMatchPerformance` model stores detailed aggregated player statistics (batting and bowling) per match, linked to `Player` and `Match` models. This model is crucial for detailed performance tracking.
    * Relationships use `ForeignKey` with appropriate `related_name` and `on_delete` policies.
* **Data Population:** Custom Django management commands (`load_match_data.py`, `load_performance_data.py`) process provided CSVs to populate the database, handling normalization and using efficient bulk operations.
* **API Development (Django REST Framework):**
    * **Read-Only Endpoints:** For `Team`, `Venue`, `Match`, `Player` data with filtering, searching, and ordering.
    * **Player Recent Performance:** Custom endpoint `GET /api/v1/players/{player_id}/recent-performance/` returns detailed stats for a player's last 5 matches, including calculated strike rates and economy rates.
    * **Prediction Endpoints:**
        * `POST /api/v1/predict_winner/`: Accepts match context, returns winner prediction and LLM explanation.
        * `POST /api/v1/predict_score/`: Accepts match context, returns predicted first innings score.
    * **LLM Follow-up Query:** `POST /api/v1/llm-query/`: A new endpoint allowing users to ask follow-up questions about a specific prediction, providing original context to the LLM.
* **Authentication:** Token-based authentication (`rest_framework.authtoken`) secures all API endpoints.
* **API Documentation:** Automated OpenAPI 3 schema via `drf-spectacular`, accessible through Swagger UI (`/api/schema/swagger-ui/`) and ReDoc (`/api/schema/redoc/`).

### 3.3. Frontend Development (Bonus Task - React)

* **Objective:** Create an interactive and intuitive frontend for the prediction system.
* **User Interface:**
    * Modern, responsive "Bento Box" grid layout with a consistent dark theme.
    * Centralized chat interface for user interaction.
* **Key Components & Features:**
    * `PredictionInputForm.jsx`: Structured form for detailed match input.
    * `ChatTextInput.jsx`: Free-text input for chat commands.
    * `PredictionGauge.jsx`: Custom animated gauge for visualizing win probability.
    * Player performance charts (runs, strike rate, wickets, economy) using `Chart.js`, displayed within the chat or a dedicated panel.
* **Chatbot Commands:** `help`, `explain last`, `performance <Player Name or ID>`, `history <Team 1> vs <Team 2>`, and contextual follow-up questions to the LLM after a prediction.

## 4. System Architecture

The system comprises:
1.  **React Frontend:** User interface for interaction, data input, and visualization.
2.  **Django Backend:** Serves main data APIs, handles player/match data, authentication, and prediction requests.
3.  **FastAPI Backend:** Provides an alternative, high-performance set of endpoints for predictions, sharing core logic with Django.
4.  **Machine Learning Models:** Pre-trained models for winner and score prediction.
5.  **Ollama LLM Service:** Local LLM providing explanations for predictions.

The frontend communicates with the Django backend for most data operations and predictions. The FastAPI backend offers a parallel route for accessing prediction functionalities. Both backends interface with the ML models and the Ollama LLM.

*(Diagram of architecture could be beneficial here if you have one, or a textual flow)*

## 5. Setup and Installation

### Prerequisites

* Python 3.8+
* Node.js & npm (for frontend)
* PostgreSQL
* Docker (recommended for ease of deployment)
* Ollama installed and running with the `phi3:instruct` model pulled (`ollama pull phi3:instruct`).

### Backend Setup (Django & FastAPI)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FrozenSaturn/ipl_predictor.git
    cd ipl_predictor
    ```
2.  **Navigate to the backend directory** (assuming a `backend/` folder or similar structure).
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Database:**
    * Ensure PostgreSQL is running.
    * Create a database (e.g., `ipl_predictor_db`).
    * Update database settings in `settings.py` (or via environment variables if configured).
6.  **Apply database migrations:**
    ```bash
    python manage.py migrate
    ```
7.  **Load initial data:**
    *(Ensure your CSV files are in the expected location for these commands)*
    ```bash
    python manage.py load_match_data
    python manage.py load_performance_data
    ```
8.  **Pre-calculate features for ML model:**
    ```bash
    python manage.py build_features
    ```
    *(Or if `build_features.py` is a standalone script: `python src/scripts/build_features.py` - adjust path as needed)*
9.  **Train ML Models (if not pre-trained and included):**
    *(Specify how to run your training script, e.g., `python src/scripts/train_models.py` - adjust path)*
10. **Run Django Development Server:**
    ```bash
    python manage.py runserver
    ```
11. **Run FastAPI Development Server:**
    *(Navigate to FastAPI app directory if separate)*
    ```bash
    uvicorn main:app --reload # Or your specific command
    ```
12. **Generate an Auth Token (Django):**
    Open the Django shell: `python manage.py shell`
    ```python
    from django.contrib.auth.models import User
    from rest_framework.authtoken.models import Token

    # Replace 'your_username' with an existing superuser or create one
    user = User.objects.get(username='your_username')
    token, created = Token.objects.get_or_create(user=user)
    print(token.key)
    ```
    Use this token in the `Authorization: Token <your_token>` header for API requests.

### Frontend Setup (React)

1.  **Navigate to the frontend directory** (e.g., `frontend/`).
2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
3.  **Start the React development server:**
    ```bash
    npm run dev
    ```
    The application should now be accessible, typically at `http://localhost:3000`.

### Ollama Setup

Ensure Ollama is running and the `phi3:instruct` model is available. The application expects Ollama to be accessible at its default address (usually `http://localhost:11434`).

## 6. API Documentation

The Django REST Framework API documentation, generated by `drf-spectacular`, is available when the Django development server is running:

* **Swagger UI:** `http://localhost:8000/api/schema/swagger-ui/`
* **ReDoc:** `http://localhost:8000/api/schema/redoc/`

## 7. Key Achievements & Innovations

This project successfully demonstrates:
* End-to-end implementation of an IPL prediction system.
* **Dual Backend Architecture:** Core prediction services are available through both Django and FastAPI, meeting diverse integration needs.
* **Rich Data Modeling:** A detailed PostgreSQL schema captures comprehensive player and match statistics, including granular per-match player performance.
* **Explainable AI:** LLM integration provides reasoning behind predictions, enhancing transparency.
* **Advanced Frontend:**
    * A highly interactive and visually appealing React interface with a modern dark theme and Bento Box layout.
    * Novel features like the contextual LLM follow-up Q&A, animated prediction confidence gauge, and in-chat player performance visualizations significantly improve user experience and analytical depth.
* **Industry-Standard Practices:** Token-based authentication, comprehensive API documentation, and structured data loading processes.

## 8. Known Limitations & Future Work

* **Model Predictive Performance:** The current F1 Macro for winner prediction (~0.54) and R² for score prediction (<0) indicate significant room for improvement. Future work should focus on:
    * More sophisticated feature engineering (e.g., detailed player rolling stats over N games, phase-based analysis, venue-player interactions, impact of injuries/weather if data can be sourced).
    * Exploration of more complex model architectures or ensembling techniques.
    * Systematic hyperparameter optimization for all models.
* **Player Performance Metrics Prediction:** Specific player-wise performance metric prediction (runs, wickets, etc., as individual outputs) was not implemented and remains a key area for future development.
* **LLM Explanation Quality:** While functional, LLM explanations are basic. Further prompt tuning, experimenting with different Ollama models, or fine-tuning could enhance relevance and consistency.
* **Live Data & Retraining:** The system currently relies on historical data. Future enhancements could include:
    * Integration of a live data feed (related to the original Task 3).
    * Implementation of a robust model retraining pipeline to keep predictions relevant.
* **Scalability & Optimization:** While functional, further optimization for database queries, API response times under load, and ML model inference speed can be explored.
* **Caching:** Implementing caching mechanisms for frequently requested predictions or data can improve performance.
* **Comprehensive Testing:** Expanding unit and integration test coverage across all components.
