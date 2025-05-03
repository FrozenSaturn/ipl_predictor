// src/components/PredictionFormInputs.jsx
import React from "react";
import PropTypes from "prop-types";

// Component focused only on rendering the form inputs
function PredictionFormInputs({
  formData,
  handleInputChange,
  handleSubmit,
  isLoading,
}) {
  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <div className="form-grid">
        <label htmlFor="team1">Team 1:</label>
        <input
          id="team1"
          name="team1" // Add name attribute for easier handling if needed
          type="text"
          value={formData.team1}
          onChange={handleInputChange}
          required
        />

        <label htmlFor="team2">Team 2:</label>
        <input
          id="team2"
          name="team2"
          type="text"
          value={formData.team2}
          onChange={handleInputChange}
          required
        />

        <label htmlFor="city">City:</label>
        <input
          id="city"
          name="city"
          type="text"
          value={formData.city}
          onChange={handleInputChange}
          required
        />

        <label htmlFor="venue">Venue:</label>
        <input
          id="venue"
          name="venue"
          type="text"
          value={formData.venue}
          onChange={handleInputChange}
          required
        />

        <label htmlFor="tossWinner">Toss Winner:</label>
        <input
          id="tossWinner"
          name="tossWinner"
          type="text"
          value={formData.tossWinner}
          onChange={handleInputChange}
          required
        />

        <label htmlFor="tossDecision">Toss Decision:</label>
        <select
          id="tossDecision"
          name="tossDecision"
          value={formData.tossDecision}
          onChange={handleInputChange}
          required
        >
          <option value="">-- Select Decision --</option>
          <option value="bat">Bat</option>
          <option value="field">Field</option>
        </select>

        <label htmlFor="matchDate">Match Date:</label>
        <input
          id="matchDate"
          name="matchDate"
          type="date"
          value={formData.matchDate}
          onChange={handleInputChange}
          required
        />
      </div>

      <button type="submit" disabled={isLoading}>
        {isLoading ? "Predicting..." : "Get Prediction"}
      </button>
    </form>
  );
}

PredictionFormInputs.propTypes = {
  formData: PropTypes.shape({
    team1: PropTypes.string.isRequired,
    team2: PropTypes.string.isRequired,
    city: PropTypes.string.isRequired,
    venue: PropTypes.string.isRequired,
    tossWinner: PropTypes.string.isRequired,
    tossDecision: PropTypes.string.isRequired,
    matchDate: PropTypes.string.isRequired,
  }).isRequired,
  handleInputChange: PropTypes.func.isRequired,
  handleSubmit: PropTypes.func.isRequired,
  isLoading: PropTypes.bool.isRequired,
};

export default PredictionFormInputs;
