// src/components/ConfidenceBar.jsx
import React from "react";
import PropTypes from "prop-types";

// A simple component to display a confidence score as a horizontal bar
function ConfidenceBar({ score }) {
  // Ensure score is between 0 and 1, default to 0 if invalid
  const validScore = Math.max(0, Math.min(1, score || 0));
  const percentage = (validScore * 100).toFixed(1); // Calculate percentage string

  // Determine bar color based on confidence (optional)
  let barColor = "#007bff"; // Default blue
  if (validScore >= 0.75) {
    barColor = "#28a745"; // Green for high confidence
  } else if (validScore >= 0.5) {
    barColor = "#ffc107"; // Yellow for medium confidence
  } else {
    barColor = "#dc3545"; // Red for low confidence
  }

  return (
    <div className="confidence-display">
      {/* Text representation */}
      <span className="confidence-score-text">{percentage}%</span>
      {/* Visual Bar */}
      <div
        className="confidence-bar-track"
        title={`Confidence: ${percentage}%`}
      >
        <div
          className="confidence-bar-fill"
          style={{ width: `${percentage}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

// Define prop types for type checking and documentation
ConfidenceBar.propTypes = {
  /** The confidence score, expected value between 0 and 1 */
  score: PropTypes.number,
};

export default ConfidenceBar;
