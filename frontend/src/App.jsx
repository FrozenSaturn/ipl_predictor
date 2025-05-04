// src/App.jsx
import React, { useState } from 'react';
import { predictMatch, getPlayers, getPlayerRecentPerformance, predictScore } from './services/api';
import PredictionInputForm from './components/PredictionInputForm';
import ChatMessageList from './components/ChatMessageList';
import ChatTextInput from './components/ChatTextInput';
import PlayerPerformance from './components/PlayerPerformance';
import './App.css'; // Or index.css where bento/dark mode styles are defined

const SENDER_USER = 'user';
const SENDER_BOT = 'bot';

function App() {
  // --- State Variables ---
  const [messages, setMessages] = useState([
    { id: 'initial', sender: SENDER_BOT, type: 'text', content: { text: "Hello! Use the form for predictions or type commands in the chat." } }
  ]);
  const [isPredictionLoading, setIsPredictionLoading] = useState(false);
  const [lastPredictionResult, setLastPredictionResult] = useState(null);
  const [isCommandLoading, setIsCommandLoading] = useState(false);
  const [formData, setFormData] = useState({
    team1: '', team2: '', city: '', venue: '',
    tossWinner: '', tossDecision: '', matchDate: '',
  });

  // --- Handlers & Helpers ---
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({ ...prevData, [name]: value }));
  };

  const addMessage = (sender, type, content) => {
    setMessages(prevMessages => [...prevMessages, { id: `${Date.now()}-${Math.random()}`, sender, type, content }]);
  };

  const handlePredict = async (submittedFormData) => {
    const userQueryText = `Predict match: ${submittedFormData.team1} vs ${submittedFormData.team2} at ${submittedFormData.venue}...`;
    addMessage(SENDER_USER, 'text', { text: userQueryText });
    setIsPredictionLoading(true);
    setLastPredictionResult(null);
    const matchData = {
      team1: submittedFormData.team1, team2: submittedFormData.team2, city: submittedFormData.city,
      venue: submittedFormData.venue, toss_winner: submittedFormData.tossWinner,
      toss_decision: submittedFormData.tossDecision, match_date: submittedFormData.matchDate,
    };
    let predictionOutcome = null;
    let scoreOutcome = null;
    let apiError = null;
    try {
      const results = await Promise.allSettled([
        predictMatch(matchData),
        predictScore(matchData)
      ]);
      console.log("API Call Results (allSettled):", results);
      if (results[0].status === 'fulfilled') {
        predictionOutcome = results[0].value;
        setLastPredictionResult(predictionOutcome);
      } else {
        console.error("Winner Prediction Failed:", results[0].reason);
        apiError = results[0].reason;
      }
      if (results[1].status === 'fulfilled') {
        scoreOutcome = results[1].value;
        if (!scoreOutcome || typeof scoreOutcome.predicted_score === 'undefined') {
          console.warn("Score prediction succeeded but response format unexpected:", scoreOutcome);
        }
      } else {
        console.error("Score Prediction Failed:", results[1].reason);
        if (!apiError) apiError = results[1].reason;
      }
      if (predictionOutcome || scoreOutcome) {
           addMessage(SENDER_BOT, 'prediction', {
              winner: predictionOutcome?.predicted_winner,
              // Corrected: Use confidence from predictionOutcome directly
              confidence: predictionOutcome?.confidence_score,
              explanation: predictionOutcome?.explanation,
              predicted_score: scoreOutcome?.predicted_score
          });
           // Optionally clear form: setFormData({ ...initial empty state... });
      }
      if (apiError) {
          const errorMessage = apiError.response?.data ? JSON.stringify(apiError.response.data) : apiError.message;
          addMessage(SENDER_BOT, 'error', { text: `Prediction partially failed or errored: ${errorMessage}` });
      }
    } catch (error) {
      console.error("Unexpected error during prediction calls:", error);
      const errorMessage = error.response?.data ? JSON.stringify(error.response.data) : error.message;
      addMessage(SENDER_BOT, 'error', { text: `Prediction failed: ${errorMessage}` });
    } finally {
      setIsPredictionLoading(false);
    }
  };

  // Handles text submitted via ChatTextInput
  const handleTextInputSend = async (text) => {
    addMessage(SENDER_USER, 'text', { text });
    // Use original casing for potentialPlayer display, lowercase for command matching
    const potentialPlayerForDisplay = text.substring('performance '.length).trim(); // Keep original casing for messages
    const command = text.toLowerCase().trim();
    setIsCommandLoading(true);

    if (command === 'help') {
      const helpText = `Available commands:\n- help: Show this message.\n- explain: Explain the last prediction.\n- performance [Player Name or ID]: Show recent performance stats.`;
      addMessage(SENDER_BOT, 'text', { text: helpText });
      setIsCommandLoading(false);

    } else if (command === 'explain' || command === 'explain last') {
      if (lastPredictionResult && lastPredictionResult.explanation) {
        addMessage(SENDER_BOT, 'text', { text: `Explanation for the last prediction (${lastPredictionResult.predicted_winner}):\n\n${lastPredictionResult.explanation}` });
      } else {
        addMessage(SENDER_BOT, 'text', { text: "No recent prediction result available to explain." });
      }
      setIsCommandLoading(false);

    } else if (command.startsWith('performance ')) {
      // Note: potentialPlayerForDisplay holds the user's input casing
      const potentialPlayer = command.substring('performance '.length).trim(); // Use lowercase for logic/ID check
      if (!potentialPlayer) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: performance [Player Name or ID]" });
        setIsCommandLoading(false); return;
      }

      addMessage(SENDER_BOT, 'text', { text: `Looking up performance for "${potentialPlayerForDisplay}"...` });
      let playerId = null;
      let playerName = potentialPlayerForDisplay; // Default to user input casing for display name

      // Check if input is numeric (potential ID)
      if (!isNaN(parseInt(potentialPlayer))) {
        playerId = parseInt(potentialPlayer);
        // We don't necessarily know the name yet if only ID is given
        playerName = `Player ID ${playerId}`;
      } else {
        // Input is potentially a name, perform search
        try {
          // Pass the name extracted from the lowercase command for searching
          const searchParams = `?search=${encodeURIComponent(potentialPlayer)}&page_size=2`; // Fetch 2 to detect ambiguity
          console.log("App.jsx: Attempting search with params:", searchParams);
          const searchResult = await getPlayers(searchParams); // Expects direct array or {results: [...]}
          console.log("App.jsx: Search API call SUCCESSFUL. Raw response received:", JSON.stringify(searchResult, null, 2));

          let playersFound = []; // Array to hold the actual player list
          if (searchResult && Array.isArray(searchResult.results)) {
              playersFound = searchResult.results; // Handle paginated
              console.log("App.jsx: Found 'results' array:", playersFound);
          } else if (searchResult && Array.isArray(searchResult)) {
              playersFound = searchResult; // Handle direct array
              console.log("App.jsx: Found direct array response:", playersFound);
          } else {
              console.log("App.jsx: Response structure unexpected or no results property found.");
          }

          console.log("App.jsx: playersFound array length:", playersFound.length);

          // Process the found players array
          if (playersFound.length === 1) {
              playerId = playersFound[0].id;
              playerName = playersFound[0].name; // Use exact name from DB for future messages
              addMessage(SENDER_BOT, 'text', { text: `Found player: ${playerName} (ID: ${playerId}). Fetching performance...` });
          } else if (playersFound.length > 1) {
              addMessage(SENDER_BOT, 'error', { text: `Multiple players found for "${potentialPlayerForDisplay}". Please use Player ID or be more specific.` });
              setIsCommandLoading(false); return; // Stop processing
          } else { // Length is 0
              addMessage(SENDER_BOT, 'error', { text: `Player "${potentialPlayerForDisplay}" not found.` });
              setIsCommandLoading(false); return; // Stop processing
          }
        } catch (searchError) {
          console.error("App.jsx: Error caught during player search:", searchError);
          addMessage(SENDER_BOT, 'error', { text: `Error searching player "${potentialPlayerForDisplay}".` });
          setIsCommandLoading(false); return; // Stop processing
        }
      } // End of name search logic

      // --- If playerId was determined (either directly or via search), fetch performance ---
      if (playerId) {
        try {
          const performance = await getPlayerRecentPerformance(playerId);
          if (performance?.length > 0) {
            // Send data to Message component for chart rendering
            addMessage(SENDER_BOT, 'performanceChart', {
              playerName: playerName, // Use name found/constructed
              performanceData: performance
            });
          } else {
            addMessage(SENDER_BOT, 'text', { text: `No recent performance data found for ${playerName}.` });
          }
        } catch (perfError) {
           const errorMessage = perfError.response?.data ? JSON.stringify(perfError.response.data) : perfError.message;
           addMessage(SENDER_BOT, 'error', { text: `Failed to get performance for ${playerName}: ${errorMessage}` });
        }
      }
      // Make sure loading stops even if playerId wasn't determined (errors handled above return)
      setIsCommandLoading(false);

    } else {
      addMessage(SENDER_BOT, 'text', { text: "Sorry, I didn't understand that. Try 'help'." });
      setIsCommandLoading(false);
    }
  }; // End handleTextInputSend

  // --- Render ---
  return (
    <div className="App bento-layout">
      <div className="bento-box input-controls">
        <div className="control-section">
          <h2>Match Prediction</h2>
          <PredictionInputForm
            formData={formData}
            handleInputChange={handleInputChange}
            onSubmitPrediction={handlePredict} // Correct prop name used here
            isLoading={isPredictionLoading}
          />
        </div>
        <hr className="bento-divider" />
        <div className="control-section">
          <PlayerPerformance />
        </div>
      </div>

      <div className="bento-box chat-interface">
        <main className="chat-main">
          <ChatMessageList messages={messages} isLoading={isPredictionLoading || isCommandLoading} />
        </main>
        <footer className="chat-footer">
          <ChatTextInput
            onSendMessage={handleTextInputSend}
            isLoading={isPredictionLoading || isCommandLoading}
          />
        </footer>
      </div>
    </div> // End bento-layout
  );
}

export default App;
