// src/components/PredictionForm.jsx
import React, { useState } from "react";
import { predictMatch } from "../services/api";
import ConfidenceBar from "./ConfidenceBar";

function PredictionForm() {
  // --- State for Form Inputs ---
  const [team1, setTeam1] = useState("");
  const [team2, setTeam2] = useState("");
  const [city, setCity] = useState("");
  const [venue, setVenue] = useState(""); // ADDED venue state
  const [tossWinner, setTossWinner] = useState("");
  const [tossDecision, setTossDecision] = useState(""); // Default to empty or 'bat'/'field'
  const [matchDate, setMatchDate] = useState(""); // Format will be YYYY-MM-DD from input type="date"

  // --- State for API Interaction ---
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- Form Submission Handler ---
  const handleSubmit = async (event) => {
    event.preventDefault(); // Prevent default browser form submission
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    // --- Retrieve Auth Token ---
    const token = localStorage.getItem("authToken");
    if (!token) {
      setError("Authentication token not found. Please login or set token.");
      setIsLoading(false);
      return;
    }
    // ---

    // --- Prepare Data Payload for API ---
    // Use the exact field names expected by the Django serializer
    const matchData = {
      team1: team1,
      team2: team2,
      city: city,
      venue: venue, // ADDED venue field to payload
      toss_winner: tossWinner,
      toss_decision: tossDecision,
      match_date: matchDate,
      // Add any other fields your specific API endpoint might require
    };

    // --- Call API ---
    try {
      console.log("Sending prediction request with data:", matchData); // Log data being sent
      const result = await predictMatch(matchData);
      console.log("Received prediction result:", result); // Log successful result
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction API call failed:", err); // Log the full error
      const backendError = err.response?.data
        ? JSON.stringify(err.response.data)
        : err.message;
      setError(`Failed to fetch prediction: ${backendError}`);
    } finally {
      setIsLoading(false); // Ensure loading indicator stops
    }
  };

  // --- Render Component ---
  return (
    <div className="prediction-container">
      <h2>IPL Match Prediction</h2>

      {/* Temporary reminder/instruction for setting the token */}
      <p
        style={{
          fontSize: "0.8em",
          color: "grey",
          border: "1px solid #eee",
          padding: "5px",
        }}
      >
        <strong>Note:</strong> Ensure you have set the auth token in
        localStorage.
        <br />
        Example (run in browser console): <br />
        <code>localStorage.setItem('authToken', 'YOUR_ACTUAL_TOKEN');</code>
      </p>

      <form onSubmit={handleSubmit} className="prediction-form">
        {" "}
        {/* Added className */}
        {/* Remove inline style from this div */}
        <div className="form-grid">
          {" "}
          {/* Added className */}
          {/* Labels and Inputs remain the same */}
          <label htmlFor="team1">Team 1:</label>
          <input
            id="team1"
            type="text"
            value={team1}
            onChange={(e) => setTeam1(e.target.value)}
            required
          />
          <label htmlFor="team2">Team 2:</label>
          <input
            id="team2"
            type="text"
            value={team2}
            onChange={(e) => setTeam2(e.target.value)}
            required
          />
          <label htmlFor="city">City:</label>
          <input
            id="city"
            type="text"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            required
          />
          <label htmlFor="venue">Venue:</label>
          <input
            id="venue"
            type="text"
            value={venue}
            onChange={(e) => setVenue(e.target.value)}
            required
          />
          <label htmlFor="tossWinner">Toss Winner:</label>
          <input
            id="tossWinner"
            type="text"
            value={tossWinner}
            onChange={(e) => setTossWinner(e.target.value)}
            required
          />
          <label htmlFor="tossDecision">Toss Decision:</label>
          <select
            id="tossDecision"
            value={tossDecision}
            onChange={(e) => setTossDecision(e.target.value)}
            required
          >
            <option value="">-- Select Decision --</option>
            <option value="bat">Bat</option>
            <option value="field">Field</option>
          </select>
          <label htmlFor="matchDate">Match Date:</label>
          <input
            id="matchDate"
            type="date"
            value={matchDate}
            onChange={(e) => setMatchDate(e.target.value)}
            required
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {" "}
          {/* Removed inline style */}
          {isLoading ? "Predicting..." : "Get Prediction"}
        </button>
      </form>

      {/* Display Area for Loading, Error, or Result */}
      <div className="results-section">
        {isLoading && <p>Loading prediction...</p>}
        {error && <p className="error-message">Error: {error}</p>}
        {predictionResult && (
          <div className="prediction-result">
            <h3>Prediction Result:</h3>
            {/* Display Winner */}
            <p>
              Predicted Winner:{" "}
              <strong>{predictionResult.predicted_winner}</strong>
            </p>

            {/* Display Confidence Label */}
            <p style={{ marginBottom: "0px" }}>Confidence:</p>
            {/* Render Confidence Bar if score exists */}
            {typeof predictionResult.confidence === "number" ? (
              <ConfidenceBar score={predictionResult.confidence} />
            ) : (
              <p>N/A</p> /* Handle case where score is missing/invalid */
            )}

            {/* Display Explanation */}
            <p>Explanation:</p>
            <pre className="explanation-box">
              {predictionResult.explanation}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionForm;
