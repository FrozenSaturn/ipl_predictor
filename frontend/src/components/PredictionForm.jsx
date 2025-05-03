// src/components/PredictionForm.jsx
import React, { useState } from "react";
import { predictMatch } from "../services/api"; // Ensure path is correct
import PredictionFormInputs from "./PredictionFormInputs"; // Import new component
import PredictionResults from "./PredictionResults"; // Import new component

function PredictionForm() {
  // --- State for Form Inputs (now consolidated) ---
  const [formData, setFormData] = useState({
    team1: "",
    team2: "",
    city: "",
    venue: "",
    tossWinner: "",
    tossDecision: "",
    matchDate: "",
  });

  // --- State for API Interaction ---
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- Generic Input Change Handler ---
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // --- Form Submission Handler ---
  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    const token = localStorage.getItem("authToken");
    if (!token) {
      setError("Authentication token not found. Please login or set token.");
      setIsLoading(false);
      return;
    }

    // Prepare data payload from state
    const matchData = {
      team1: formData.team1,
      team2: formData.team2,
      city: formData.city,
      venue: formData.venue,
      toss_winner: formData.tossWinner,
      toss_decision: formData.tossDecision,
      match_date: formData.matchDate,
    };

    try {
      console.log("Sending prediction request with data:", matchData);
      const result = await predictMatch(matchData);
      console.log("Received prediction result:", result);
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction API call failed:", err);
      const backendError = err.response?.data
        ? JSON.stringify(err.response.data)
        : err.message;
      setError(`Failed to fetch prediction: ${backendError}`);
    } finally {
      setIsLoading(false);
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

      {/* Render the Input Form Component */}
      <PredictionFormInputs
        formData={formData}
        handleInputChange={handleInputChange}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
      />

      {/* Render the Results Display Component */}
      <div className="results-section">
        <PredictionResults
          isLoading={isLoading}
          error={error}
          result={predictionResult}
        />
      </div>
    </div>
  );
}

// No need for PropTypes here if it's the top-level component, but can add if desired.

export default PredictionForm;
