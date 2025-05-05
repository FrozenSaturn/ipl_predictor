// src/App.jsx
import React, { useState } from 'react';
import { predictMatch, getPlayers, getPlayerRecentPerformance, predictScore, getMatchHistory } from './services/api';
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
              confidence: predictionOutcome?.confidence, // Corrected key from previous step
              explanation: predictionOutcome?.explanation,
              predicted_score: scoreOutcome?.predicted_score,
              team1Name: submittedFormData.team1,
              team2Name: submittedFormData.team2

          });
           // Optionally clear form: setFormData({ team1:'', team2:'', ... });
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
    const commandText = text.trim(); // Keep original casing for display
    const command = commandText.toLowerCase(); // Use lowercase for matching
    setIsCommandLoading(true); // Set loading true for command processing

    // --- Corrected Command Parsing Logic ---
    if (command === 'help') {
      const helpText = `Available commands:\n- help\n- explain\n- performance [Player Name/ID]\n- history [Team 1] vs [Team 2]`;
      addMessage(SENDER_BOT, 'text', { text: helpText });
      setIsCommandLoading(false); // Stop loading

    } else if (command === 'explain' || command === 'explain last') {
      if (lastPredictionResult && lastPredictionResult.explanation) {
        addMessage(SENDER_BOT, 'text', { text: `Explanation for the last prediction (${lastPredictionResult.predicted_winner}):\n\n${lastPredictionResult.explanation}` });
      } else {
        addMessage(SENDER_BOT, 'text', { text: "No recent prediction result available to explain." });
      }
      setIsCommandLoading(false); // Stop loading

    } else if (command.startsWith('performance ')) {
      const potentialPlayerForDisplay = commandText.substring('performance '.length).trim();
      const potentialPlayer = potentialPlayerForDisplay.toLowerCase(); // Use lowercase for logic/ID check

      if (!potentialPlayer) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: performance [Player Name or ID]" });
        setIsCommandLoading(false); return;
      }

      addMessage(SENDER_BOT, 'text', { text: `Looking up performance for "${potentialPlayerForDisplay}"...` });
      let playerId = null;
      let playerName = potentialPlayerForDisplay;

      if (!isNaN(parseInt(potentialPlayer))) { // Check if it's a number (ID)
        playerId = parseInt(potentialPlayer);
        playerName = `Player ID ${playerId}`; // Use ID as placeholder name
      } else { // It's likely a name, search for it
        try {
          const searchParams = `?search=${encodeURIComponent(potentialPlayerForDisplay)}&page_size=2`; // Search using original casing maybe? Or lowercase 'potentialPlayer'? Let's try original casing first.
          const searchResult = await getPlayers(searchParams);
          let playersFound = [];
          if (searchResult && Array.isArray(searchResult.results)) { playersFound = searchResult.results; }
          else if (searchResult && Array.isArray(searchResult)) { playersFound = searchResult; }

          if (playersFound.length === 1) {
            playerId = playersFound[0].id;
            playerName = playersFound[0].name; // Use exact name from DB
            addMessage(SENDER_BOT, 'text', { text: `Found player: ${playerName} (ID: ${playerId}). Fetching performance...` });
          } else if (playersFound.length > 1) {
            addMessage(SENDER_BOT, 'error', { text: `Multiple players found for "${potentialPlayerForDisplay}". Use ID or be more specific.` });
            setIsCommandLoading(false); return;
          } else {
            addMessage(SENDER_BOT, 'error', { text: `Player "${potentialPlayerForDisplay}" not found.` });
            setIsCommandLoading(false); return;
          }
        } catch (searchError) {
          console.error("App.jsx: Error caught during player search:", searchError);
          addMessage(SENDER_BOT, 'error', { text: `Error searching player "${potentialPlayerForDisplay}".` });
          setIsCommandLoading(false); return;
        }
      } // End name search logic

      // If playerId determined, fetch performance
      if (playerId) {
        try {
          const performance = await getPlayerRecentPerformance(playerId);
          if (performance?.length > 0) {
            addMessage(SENDER_BOT, 'performanceChart', { playerName: playerName, performanceData: performance });
          } else {
            addMessage(SENDER_BOT, 'text', { text: `No recent performance data found for ${playerName}.` });
          }
        } catch (perfError) {
          const errorMessage = perfError.response?.data ? JSON.stringify(perfError.response.data) : perfError.message;
          addMessage(SENDER_BOT, 'error', { text: `Failed to get performance for ${playerName}: ${errorMessage}` });
        }
      }
      setIsCommandLoading(false); // Stop loading after performance logic completes

    // --- HISTORY COMMAND BLOCK - MOVED TO CORRECT LEVEL ---
    } else if (command.startsWith('history ')) {
      const teamsString = commandText.substring('history '.length).trim();
      const teams = teamsString.split(/ vs /i); // Split by ' vs ' case-insensitively

      if (teams.length !== 2 || !teams[0] || !teams[1]) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: history [Team 1] vs [Team 2]" });
        setIsCommandLoading(false); return;
      }

      // --- Use the names directly parsed from the user's command ---
      const requestedTeam1 = teams[0].trim();
      const requestedTeam2 = teams[1].trim();
      // --- End change ---

      addMessage(SENDER_BOT, 'text', { text: `Workspaceing match history for ${requestedTeam1} vs ${requestedTeam2}...` });

      try {
        // API call remains the same (still potentially returns extra matches)
        const historyResult = await getMatchHistory(requestedTeam1, requestedTeam2);

        let matches = [];
        if (historyResult && Array.isArray(historyResult.results)) { matches = historyResult.results; }
        else if (historyResult && Array.isArray(historyResult)) { matches = historyResult; }

        console.log("App.jsx: Raw matches received from API for history:", JSON.stringify(matches, null, 2));

        if (matches.length === 0) {
           // This case might still happen if the broad search finds nothing at all
           addMessage(SENDER_BOT, 'text', { text: `No match history found containing both ${requestedTeam1} and ${requestedTeam2}.` });
        } else {
            let team1Wins = 0;
            let team2Wins = 0;
            let drawsOrNR = 0;
            let actualMatchesPlayed = 0; // Count only relevant matches

            matches.forEach(match => {
                const matchTeam1Name = match.team1?.name;
                const matchTeam2Name = match.team2?.name;
                const winnerName = match.winner?.name;

                // --- Check if this match involves the two TEAMS REQUESTED BY THE USER ---
                const involvesRequestedTeams =
                    (matchTeam1Name === requestedTeam1 && matchTeam2Name === requestedTeam2) ||
                    (matchTeam1Name === requestedTeam2 && matchTeam2Name === requestedTeam1);
                // --- End change ---

                console.log(`Processing Match ID: ${match.id}, T1: ${matchTeam1Name}, T2: ${matchTeam2Name}, Winner: ${winnerName}, InvolvesRequested? ${involvesRequestedTeams}`);

                // --- Only count if it's a direct H2H match ---
                if (involvesRequestedTeams) {
                    actualMatchesPlayed++; // Increment count of relevant matches

                    // Count wins based on REQUESTED team names
                    if (!winnerName && match.result !== 'tie') {
                         drawsOrNR++;
                    } else if (winnerName === requestedTeam1) { // Compare winner to requestedTeam1
                         team1Wins++;
                    } else if (winnerName === requestedTeam2) { // Compare winner to requestedTeam2
                         team2Wins++;
                    } else { // Includes actual ties or cases where winner name doesn't match (e.g., old data)
                         drawsOrNR++;
                    }
                } else {
                    console.warn("Match result filtered out as teams didn't match requested H2H query:", match);
                }
            });

            const totalPlayed = actualMatchesPlayed; // Use the count of correctly filtered matches
            console.log(`App.jsx: Finished processing. Total Played (filtered): ${totalPlayed}, ${requestedTeam1} Wins: ${team1Wins}, ${requestedTeam2} Wins: ${team2Wins}, Draw/NR: ${drawsOrNR}`);

            if (totalPlayed === 0) {
                 // This now means the API returned matches, but NONE were between the two requested teams
                 addMessage(SENDER_BOT, 'text', { text: `Found related matches, but none directly between ${requestedTeam1} and ${requestedTeam2}.` });
            } else {
                // --- Use REQUESTED names in the final summary ---
                addMessage(SENDER_BOT, 'historySummary', {
                  team1Name: requestedTeam1, // Use names user requested
                  team2Name: requestedTeam2,
                  played: totalPlayed,
                  team1Wins: team1Wins,
                  team2Wins: team2Wins,
                  drawsOrNR: drawsOrNR,
              });

            }
        }
      } catch (histError) { /* ... error handling ... */ }
      finally { setIsCommandLoading(false); }


    } else {
      // Default case for unknown commands
      addMessage(SENDER_BOT, 'text', { text: "Sorry, I didn't understand that. Try 'help'." });
      setIsCommandLoading(false); // Stop loading
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
            onSubmitPrediction={handlePredict}
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
