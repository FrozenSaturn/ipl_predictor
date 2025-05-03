// src/components/PredictionResults.jsx
import React from "react";
import PropTypes from "prop-types";
import ConfidenceBar from "./ConfidenceBar"; // Assuming ConfidenceBar is in the same directory

function PredictionResults({ isLoading, error, result }) {
  if (isLoading) {
    return <p>Loading prediction...</p>;
  }

  if (error) {
    return <p className="error-message">Error: {error}</p>;
  }

  if (!result) {
    return null; // Render nothing if there's no result yet (and not loading/error)
  }

  // Render the results if available
  return (
    <div className="prediction-result">
      <h3>Prediction Result:</h3>
      <p>
        Predicted Winner: <strong>{result.predicted_winner}</strong>
      </p>

      <p style={{ marginBottom: "0px" }}>Confidence:</p>
      {typeof result.confidence === "number" ? (
        <ConfidenceBar score={result.confidence} />
      ) : (
        <p>N/A</p>
      )}

      <p>Explanation:</p>
      <pre className="explanation-box">{result.explanation}</pre>
    </div>
  );
}

PredictionResults.propTypes = {
  isLoading: PropTypes.bool.isRequired,
  error: PropTypes.string, // Error message string, null if no error
  result: PropTypes.shape({
    // Object structure of the expected result
    predicted_winner: PropTypes.string,
    confidence_score: PropTypes.number,
    explanation: PropTypes.string,
    // Add other expected fields if necessary
  }), // Result can be null initially
};

export default PredictionResults;
