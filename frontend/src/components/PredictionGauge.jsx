// src/components/PredictionGauge.jsx
import React from 'react';
import PropTypes from 'prop-types';

// Define fixed rotation limits
const MIN_ROTATION = -90; // Fully Team 1
const MAX_ROTATION = 90;  // Fully Team 2
const CENTER_ROTATION = 0; // 50/50

function PredictionGauge({ score, winner, team1Name, team2Name }) {
  // Validate score (0 to 1)
  const validScore = Math.max(0, Math.min(1, score || 0.5)); // Default to 0.5 if invalid

  let rotation = CENTER_ROTATION;

  // Determine rotation based on winner and score
  if (winner && winner === team1Name) {
    // Score represents confidence in Team 1 winning.
    // Map 0.5 confidence -> 0deg, 1.0 confidence -> -90deg
    // Map score range [0.5, 1.0] to rotation range [0, -90]
    rotation = (0.5 - validScore) * 180; // (0.5-0.5)*180=0; (0.5-1.0)*180 = -90
  } else if (winner && winner === team2Name) {
    // Score represents confidence in Team 2 winning.
    // Map 0.5 confidence -> 0deg, 1.0 confidence -> +90deg
    // Map score range [0.5, 1.0] to rotation range [0, +90]
    rotation = (validScore - 0.5) * 180; // (0.5-0.5)*180=0; (1.0-0.5)*180 = +90
  } else {
    // If winner is null, unclear, or doesn't match teams, keep needle centered
    rotation = CENTER_ROTATION;
    console.warn("Gauge centering needle due to missing/mismatched winner or low confidence implied.");
    // Note: Confidence score < 0.5 technically shouldn't happen if 'winner' is set,
    // but this logic handles it by defaulting to center.
  }

  // Clamp rotation to limits
  const finalRotation = Math.max(MIN_ROTATION, Math.min(MAX_ROTATION, rotation));
  const percentage = (validScore * 100).toFixed(1);

  return (
    <div className="gauge-container" title={`Confidence: ${percentage}% towards ${winner || 'undecided'}`}>
      <div className="gauge-semi-circle">
        {/* Optional: Add color segments or ticks here */}
      </div>
      <div
        className="gauge-needle"
        style={{ transform: `rotate(${finalRotation}deg)` }}
      ></div>
      <div className="gauge-center-pivot"></div> {/* Optional: Small circle at needle base */}
      <div className="gauge-value-text">{percentage}%</div> {/* Display percentage */}
    </div>
  );
}

PredictionGauge.propTypes = {
  score: PropTypes.number, // Confidence score (0-1)
  winner: PropTypes.string, // Predicted winner name (must match team1Name or team2Name)
  team1Name: PropTypes.string, // Name of team 1 (for orientation)
  team2Name: PropTypes.string, // Name of team 2 (for orientation)
};

export default PredictionGauge;
