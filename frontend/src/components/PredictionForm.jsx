// src/components/PredictionForm.jsx
import React, { useState } from 'react';
import PropTypes from 'prop-types';
// No need to import API service here anymore
// No need for PredictionResults/ConfidenceBar imports here

// Renamed to reflect its new role
function PredictionInputForm({ formData, handleInputChange, handleSubmit, isPredictionLoading }) { // Receive props
  // State just for the form fields
  const [formData, setFormData] = useState({
    team1: '',
    team2: '',
    city: '',
    venue: '',
    tossWinner: '',
    tossDecision: '',
    matchDate: '',
  });

  // Generic Input Change Handler
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle submission by calling the callback prop
  const handleSubmit = (event) => {
    event.preventDefault();
    // Pass the current form data up to the parent
    onPredict(formData);
    // Optionally clear the form after submission?
    // setFormData({ team1: '', team2: '', city: '', ... });
  };

  return (
    // Removed className="prediction-container" - layout handled by App.jsx
    <div>
       {/* Optional: Add a title or instruction */}
       {/* <h4>Enter Match Details:</h4> */}

      {/* Use the renamed handleSubmit */}
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-grid">
          {/* Use handleInputChange for all inputs */}
          {/* Add 'name' attribute matching state keys */}
          <label htmlFor="team1">Team 1:</label>
          <input id="team1" name="team1" type="text" value={formData.team1} onChange={handleInputChange} required disabled={isPredictionLoading} />

          <label htmlFor="team2">Team 2:</label>
          <input id="team2" name="team2" type="text" value={formData.team2} onChange={handleInputChange} required disabled={isPredictionLoading} />

          <label htmlFor="city">City:</label>
          <input id="city" name="city" type="text" value={formData.city} onChange={handleInputChange} required disabled={isPredictionLoading} />

          <label htmlFor="venue">Venue:</label>
          <input id="venue" name="venue" type="text" value={formData.venue} onChange={handleInputChange} required disabled={isPredictionLoading} />

          <label htmlFor="tossWinner">Toss Winner:</label>
          <input id="tossWinner" name="tossWinner" type="text" value={formData.tossWinner} onChange={handleInputChange} required disabled={isPredictionLoading} />

          <label htmlFor="tossDecision">Toss Decision:</label>
          <select id="tossDecision" name="tossDecision" value={formData.tossDecision} onChange={handleInputChange} required disabled={isPredictionLoading} >
            <option value="">-- Select --</option>
            <option value="bat">Bat</option>
            <option value="field">Field</option>
          </select>

          <label htmlFor="matchDate">Match Date:</label>
          <input id="matchDate" name="matchDate" type="date" value={formData.matchDate} onChange={handleInputChange} required disabled={isPredictionLoading} />
        </div>

        {/* Disable button while parent indicates loading */}
        <button type="submit" disabled={isPredictionLoading}>
          {isPredictionLoading ? 'Getting Prediction...' : 'Send Prediction Request'}
        </button>
      </form>

      {/* Result display is now handled by the parent App component */}

    </div>

  );
}

PredictionInputForm.propTypes = {
  formData: PropTypes.object.isRequired,
  handleInputChange: PropTypes.func.isRequired,
  handleSubmit: PropTypes.func.isRequired, // Expecting handleSubmit
  isPredictionLoading: PropTypes.bool.isRequired,
};

// Rename the export
export default PredictionInputForm;
