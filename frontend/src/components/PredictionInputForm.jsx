// src/components/PredictionInputForm.jsx
import React, { useState } from 'react';
import PropTypes from 'prop-types';

// This form manages its own input state and calls a prop function on submit
function PredictionInputForm({ onSubmitPrediction, isLoading }) {
  // Internal state to manage the values of the form fields
  const [formData, setFormData] = useState({
    team1: '',
    team2: '',
    city: '',
    venue: '',
    tossWinner: '',
    tossDecision: '',
    matchDate: '',
  });

  // Handles changes in any input field
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handles the form submission
  const handleSubmit = (event) => {
    event.preventDefault(); // Prevent default browser submission
    console.log("PredictionInputForm: Form submitted with data:", formData);
    // Call the callback function passed from App.jsx with the current form data
    onSubmitPrediction(formData);

    // Optionally clear the form fields after submission
    // setFormData({
    //   team1: '', team2: '', city: '', venue: '',
    //   tossWinner: '', tossDecision: '', matchDate: ''
    // });
  };

  return (
    // No outer container needed if App.jsx footer provides padding etc.
    <form onSubmit={handleSubmit} className="prediction-form"> {/* Use internal handleSubmit */}
      <div className="form-grid">
        {/* Team 1 */}
        <label htmlFor="chat-team1">Team 1:</label>
        <input id="chat-team1" name="team1" type="text" value={formData.team1} onChange={handleInputChange} required disabled={isLoading} />

        {/* Team 2 */}
        <label htmlFor="chat-team2">Team 2:</label>
        <input id="chat-team2" name="team2" type="text" value={formData.team2} onChange={handleInputChange} required disabled={isLoading} />

        {/* City */}
        <label htmlFor="chat-city">City:</label>
        <input id="chat-city" name="city" type="text" value={formData.city} onChange={handleInputChange} required disabled={isLoading} />

        {/* Venue */}
        <label htmlFor="chat-venue">Venue:</label>
        <input id="chat-venue" name="venue" type="text" value={formData.venue} onChange={handleInputChange} required disabled={isLoading} />

        {/* Toss Winner */}
        <label htmlFor="chat-tossWinner">Toss Winner:</label>
        <input id="chat-tossWinner" name="tossWinner" type="text" value={formData.tossWinner} onChange={handleInputChange} required disabled={isLoading} />

        {/* Toss Decision */}
        <label htmlFor="chat-tossDecision">Toss Decision:</label>
        <select id="chat-tossDecision" name="tossDecision" value={formData.tossDecision} onChange={handleInputChange} required disabled={isLoading} >
          <option value="">-- Select --</option>
          <option value="bat">Bat</option>
          <option value="field">Field</option>
        </select>

        {/* Match Date */}
        <label htmlFor="chat-matchDate">Match Date:</label>
        <input id="chat-matchDate" name="matchDate" type="date" value={formData.matchDate} onChange={handleInputChange} required disabled={isLoading} />
      </div>

      {/* Submit Button */}
      <button type="submit" disabled={isLoading} style={{marginTop:'10px'}}> {/* Added slight margin */}
        {isLoading ? 'Getting Prediction...' : 'Send Prediction Request'}
      </button>
    </form>
  );
}

// Define expected prop types
PredictionInputForm.propTypes = {
  // Expects a function to call when form is submitted
  onSubmitPrediction: PropTypes.func.isRequired,
  // Expects a boolean indicating if parent is loading
  isLoading: PropTypes.bool.isRequired,
};

export default PredictionInputForm;
