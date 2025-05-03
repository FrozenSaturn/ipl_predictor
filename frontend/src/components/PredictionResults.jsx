// src/components/PredictionResults.jsx
import React from 'react';
import PropTypes from 'prop-types';
import ConfidenceBar from './ConfidenceBar'; // Assuming ConfidenceBar is in the same directory

function PredictionResults({ isLoading, error, result }) {
  // 1. Handle Loading State
  if (isLoading) {
    // Display loading message specifically for prediction results
    return <p>Loading prediction results...</p>;
  }

  // 2. Handle Error State
  if (error) {
    // Display error related to fetching prediction
    return <p className="error-message">Prediction Error: {error}</p>;
  }

  // 3. Handle No Result State (after loading/error checks)
  if (!result) {
    // Nothing to display yet (initial state or cleared results)
    return null;
  }

  // 4. Display Results (if result object exists)
  return (
    <div className="prediction-result">
      <h3>Prediction Result:</h3>

      {/* Display Winner */}
      <p>Predicted Winner: <strong>{result.predicted_winner || 'N/A'}</strong></p>

      {/* Display Confidence Label and Bar */}
      <p style={{ marginBottom: '0px' }}>Confidence:</p>
      {typeof result.confidence === 'number' ? (
        <ConfidenceBar score={result.confidence} />
      ) : (
        <p>N/A</p> /* Handle case where score is missing/invalid */
      )}

      {/* Display Explanation - Conditionally */}
      {/* Check if result.explanation is present and not empty */}
      {result.explanation && result.explanation.trim() !== '' ? (
        <>
          <p style={{ marginTop: '15px' }}>Explanation:</p>
          <pre className="explanation-box">
            {result.explanation}
          </pre>
        </>
      ) : (
        // Render this if explanation is null, undefined, or empty string
        <p style={{ marginTop: '15px', fontStyle: 'italic' }}>
          No explanation provided.
        </p>
      )}
    </div>
  );
}

// Updated PropTypes to clarify structure (explanation is optional string)
PredictionResults.propTypes = {
  isLoading: PropTypes.bool.isRequired,
  error: PropTypes.string,
  result: PropTypes.shape({
    predicted_winner: PropTypes.string,
    confidence_score: PropTypes.number,
    explanation: PropTypes.string, // Explanation might be missing
    // Add other expected fields if necessary
  }),
};

export default PredictionResults;
