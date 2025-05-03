// src/App.jsx
import React, { useState } from 'react';
import { predictMatch, getPlayers, getPlayerRecentPerformance } from './services/api';
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
  const [isCommandLoading, setIsCommandLoading] = useState(false); // Separate loading for text commands
  const [formData, setFormData] = useState({ // State for the prediction input form
    team1: '', team2: '', city: '', venue: '',
    tossWinner: '', tossDecision: '', matchDate: '',
  });

  // --- Handlers & Helpers ---

  // Manages controlled inputs for PredictionInputForm
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({ ...prevData, [name]: value }));
  };

  // Adds a message object to the chat history
  const addMessage = (sender, type, content) => {
    setMessages(prevMessages => [...prevMessages, { id: `${Date.now()}-${Math.random()}`, sender, type, content }]);
  };

  // Handles submission from PredictionInputForm
  const handlePredict = async (submittedFormData) => {
    const userQueryText = `Predict match: ${submittedFormData.team1} vs ${submittedFormData.team2} at ${submittedFormData.venue}...`; // Shorter summary
    addMessage(SENDER_USER, 'text', { text: userQueryText });
    setIsPredictionLoading(true);
    setLastPredictionResult(null);

    const matchData = {
      team1: submittedFormData.team1, team2: submittedFormData.team2, city: submittedFormData.city,
      venue: submittedFormData.venue, toss_winner: submittedFormData.tossWinner,
      toss_decision: submittedFormData.tossDecision, match_date: submittedFormData.matchDate,
    };

    try {
      const predictionResult = await predictMatch(matchData);
      setLastPredictionResult(predictionResult); // Store result
      addMessage(SENDER_BOT, 'prediction', {
          winner: predictionResult.predicted_winner,
          confidence: predictionResult.confidence_score,
          explanation: predictionResult.explanation
      });
      // Optionally clear form: setFormData({ ...initial empty state... });
    } catch (error) {
      const errorMessage = error.response?.data ? JSON.stringify(error.response.data) : error.message;
      addMessage(SENDER_BOT, 'error', { text: `Prediction failed: ${errorMessage}` });
    } finally {
      setIsPredictionLoading(false);
    }
  };

  // Handles text submitted via ChatTextInput
  const handleTextInputSend = async (text) => {
    addMessage(SENDER_USER, 'text', { text });
    const command = text.toLowerCase().trim();
    setIsCommandLoading(true); // Indicate bot is processing command

    // Command Parsing Logic
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
      const potentialPlayer = command.substring('performance '.length).trim();
      if (!potentialPlayer) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: performance [Player Name or ID]" });
        setIsCommandLoading(false);
        return;
      }

      addMessage(SENDER_BOT, 'text', { text: `Looking up performance for "${potentialPlayer}"...` });
      let playerId = null;
      let playerName = potentialPlayer;

      if (!isNaN(parseInt(potentialPlayer))) {
        playerId = parseInt(potentialPlayer);
      } else {
        try {
          const searchResult = await getPlayers(`?search=${encodeURIComponent(playerName)}&page_size=1`); // Use getPlayers for search
          if (searchResult?.results?.length === 1) {
            playerId = searchResult.results[0].id;
            playerName = searchResult.results[0].name;
          } else if (searchResult?.results?.length > 1) {
            addMessage(SENDER_BOT, 'error', { text: `Multiple players found for "${potentialPlayer}". Use ID or be more specific.` });
            setIsCommandLoading(false); return;
          } else {
            addMessage(SENDER_BOT, 'error', { text: `Player "${potentialPlayer}" not found.` });
            setIsCommandLoading(false); return;
          }
        } catch (searchError) {
          addMessage(SENDER_BOT, 'error', { text: `Error searching player "${potentialPlayer}".` });
          setIsCommandLoading(false); return;
        }
      }

      if (playerId) {
        try {
          const performance = await getPlayerRecentPerformance(playerId);
          if (performance?.length > 0) {
            const perfSummary = performance.map((p, index) =>
                `Match ${performance.length - index} (${new Date(p.match_date).toLocaleDateString('en-GB',{day:'2-digit',month:'short'})} @ ${p.match_venue}): ` +
                (p.runs_scored !== null ? `Bat: ${p.runs_scored}(${p.balls_faced}), SR: ${p.strike_rate?.toFixed(1)} ` : '') +
                (p.balls_bowled !== null ? `Bowl: ${p.wickets_taken}/${p.runs_conceded}(${(p.balls_bowled/6).toFixed(1)} ov), Econ: ${p.economy_rate?.toFixed(1)}` : '')
            ).join('\n');
            addMessage(SENDER_BOT, 'text', { text: `Recent Performance for ${playerName}:\n${perfSummary}` });
          } else {
            addMessage(SENDER_BOT, 'text', { text: `No recent performance data for ${playerName} (ID: ${playerId}).` });
          }
        } catch (perfError) {
           const errorMessage = perfError.response?.data ? JSON.stringify(perfError.response.data) : perfError.message;
           addMessage(SENDER_BOT, 'error', { text: `Failed to get performance for ${playerName}: ${errorMessage}` });
        }
      }
      setIsCommandLoading(false);

    } else {
      addMessage(SENDER_BOT, 'text', { text: "Sorry, I didn't understand that. Try 'help'." });
      setIsCommandLoading(false); // Ensure loading stops for unknown commands
    }
  };

  // --- Render ---
  return (
    // Apply bento-layout class for CSS Grid
    <div className="App bento-layout">

      {/* --- Left Column: Inputs/Controls --- */}
      <div className="bento-box input-controls">
        <div className="control-section">
          <h2>Match Prediction</h2>
          <PredictionInputForm
            // Pass state and handlers for controlled form
            formData={formData}
            handleInputChange={handleInputChange}
            // Pass prediction handler as 'handleSubmit' prop
            onSubmitPrediction={handlePredict}
            isLoading={isPredictionLoading}
          />
        </div>

        <hr className="bento-divider" />

        <div className="control-section">
          {/* PlayerPerformance component renders its own title */}
          <PlayerPerformance />
          {/* Note: Selecting player here won't affect chat unless explicitly integrated */}
        </div>
      </div>

      {/* --- Right Column: Chat Interface --- */}
      <div className="bento-box chat-interface">
        {/* Optional: Add a header within the chat box if desired */}
        {/* <header className="chat-header"><h2>Chat</h2></header> */}
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
