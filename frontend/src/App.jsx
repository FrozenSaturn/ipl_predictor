import React, { useState } from 'react';
import { predictMatch, getPlayers, getPlayerRecentPerformance, predictScore, getMatchHistory, queryLLMContextual } from './services/api';
import PredictionInputForm from './components/PredictionInputForm';
import ChatMessageList from './components/ChatMessageList';
import ChatTextInput from './components/ChatTextInput';
import PlayerPerformance from './components/PlayerPerformance';
import './App.css';

const SENDER_USER = 'user';
const SENDER_BOT = 'bot';

function App() {
  // --- State Variables ---
  const [messages, setMessages] = useState([
    { id: 'initial', sender: SENDER_BOT, type: 'text', content: { text: "Hello! Use the form for predictions or type commands in the chat." } }
  ]);
  const [isPredictionLoading, setIsPredictionLoading] = useState(false);
  // lastPredictionResult stores the raw prediction result from predictMatch.
  // lastPredictionInfo is generally preferred as it holds combined results and context for follow-ups.
  const [lastPredictionResult, setLastPredictionResult] = useState(null);
  const [isCommandLoading, setIsCommandLoading] = useState(false);
  const [formData, setFormData] = useState({
    team1: '', team2: '', city: '', venue: '',
    tossWinner: '', tossDecision: '', matchDate: '',
  });
  // Stores the full context (input data) and combined results (winner, score, explanation) of the last prediction.
  // Used for 'explain' command and contextual LLM follow-up queries.
  const [lastPredictionInfo, setLastPredictionInfo] = useState(null);

  // --- Handlers & Helpers ---
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({ ...prevData, [name]: value }));
  };
  const addMessage = (sender, type, content) => {
    setMessages(prevMessages => [...prevMessages, { id: `${Date.now()}-${Math.random()}`, sender, type, content }]);
  };

  // Handles submission from PredictionInputForm
  const handlePredict = async (submittedFormData) => {
    const userQueryText = `Predict match: ${submittedFormData.team1} vs ${submittedFormData.team2} at ${submittedFormData.venue}...`;
    addMessage(SENDER_USER, 'text', { text: userQueryText });
    setIsPredictionLoading(true);
    setLastPredictionInfo(null); // Clear previous prediction context before a new prediction
    // setLastPredictionResult(null); // Clear this too if keeping both state vars

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
        // setLastPredictionResult(predictionOutcome); // Store base result if needed separately
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

      // Process and store results together if at least one succeeded
      if (predictionOutcome || scoreOutcome) {
           const resultData = {
              winner: predictionOutcome?.predicted_winner,
              confidence: predictionOutcome?.confidence,
              explanation: predictionOutcome?.explanation,
              predicted_score: scoreOutcome?.predicted_score,
              // Include team names needed by gauge in Message.jsx
              team1Name: submittedFormData.team1,
              team2Name: submittedFormData.team2
          };
           addMessage(SENDER_BOT, 'prediction', resultData);

           // --- STORE BOTH RESULT AND CONTEXT ---
           setLastPredictionInfo({
               prediction: resultData, // Store the combined results object
               context: matchData      // Store the input object used
           });
           // --- END STORE ---

      }
      if (apiError) {
          const errorMessage = apiError.response?.data ? JSON.stringify(apiError.response.data) : apiError.message;
          addMessage(SENDER_BOT, 'error', { text: `Prediction partially failed or errored: ${errorMessage}` });
          setLastPredictionInfo(null); // Clear context as prediction encountered an error
      }
    } catch (error) {
      console.error("Unexpected error during prediction calls:", error);
      const errorMessage = error.response?.data ? JSON.stringify(error.response.data) : error.message;
      addMessage(SENDER_BOT, 'error', { text: `Prediction failed: ${errorMessage}` });
      setLastPredictionInfo(null); // Clear context on any general failure
    } finally {
      setIsPredictionLoading(false);
    }
  };

  // Handles text submitted via ChatTextInput
  const handleTextInputSend = async (text) => {
    addMessage(SENDER_USER, 'text', { text });
    const commandText = text.trim();
    const command = commandText.toLowerCase();
    setIsCommandLoading(true);

    // --- Main Command Parsing Logic ---
    if (command === 'help') {
      const helpText = `Available commands:\n- help\n- explain\n- performance [Player Name/ID]\n- history [Team 1] vs [Team 2]`;
      addMessage(SENDER_BOT, 'text', { text: helpText });
      setIsCommandLoading(false);

    } else if (command === 'explain' || command === 'explain last') {
      // Explain command uses the explanation from the last stored prediction info.
      if (lastPredictionInfo?.prediction?.explanation) {
        addMessage(SENDER_BOT, 'text', { text: `Explanation for the last prediction (${lastPredictionInfo.prediction.winner || 'N/A'}):\n\n${lastPredictionInfo.prediction.explanation}` });
      } else {
        addMessage(SENDER_BOT, 'text', { text: "No recent prediction result available to explain." });
      }
      setIsCommandLoading(false);

    } else if (command.startsWith('performance ')) {
      // --- Player Performance Command Logic ---
      const potentialPlayerForDisplay = commandText.substring('performance '.length).trim();
      const potentialPlayer = potentialPlayerForDisplay.toLowerCase();

      if (!potentialPlayer) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: performance [Player Name or ID]" });
        setIsCommandLoading(false); return;
      }

      addMessage(SENDER_BOT, 'text', { text: `Looking up performance for "${potentialPlayerForDisplay}"...` });
      let playerId = null;
      let playerName = potentialPlayerForDisplay;

      if (!isNaN(parseInt(potentialPlayer))) { // Input is a number, assume it's a Player ID
        playerId = parseInt(potentialPlayer);
        playerName = `Player ID ${playerId}`;
      } else { // Input is not a number, assume it's a Player Name and search
        try {
          const searchParams = `?search=${encodeURIComponent(potentialPlayerForDisplay)}&page_size=2`;
          console.log("App.jsx: Attempting player search with params:", searchParams);
          const searchResult = await getPlayers(searchParams);
          console.log("App.jsx: Search API call SUCCESSFUL. Raw response received:", JSON.stringify(searchResult, null, 2));

          let playersFound = [];
          if (searchResult && Array.isArray(searchResult.results)) { playersFound = searchResult.results; }
          else if (searchResult && Array.isArray(searchResult)) { playersFound = searchResult; }
          else { console.log("App.jsx: Player search response structure unexpected."); }

          console.log("App.jsx: playersFound array length:", playersFound.length);

          if (playersFound.length === 1) {
            playerId = playersFound[0].id;
            playerName = playersFound[0].name;
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
      }

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
      setIsCommandLoading(false);
      // --- End Player Performance Command Logic ---
    } else if (command.startsWith('history ')) {
      // --- Match History Command Logic ---
      const teamsString = commandText.substring('history '.length).trim();
      const teams = teamsString.split(/ vs /i);

      if (teams.length !== 2 || !teams[0] || !teams[1]) {
        addMessage(SENDER_BOT, 'error', { text: "Usage: history [Team 1] vs [Team 2]" });
        setIsCommandLoading(false); return;
      }
      const requestedTeam1 = teams[0].trim();
      const requestedTeam2 = teams[1].trim();
      addMessage(SENDER_BOT, 'text', { text: `Workspaceing match history for ${requestedTeam1} vs ${requestedTeam2}...` });

      try {
        const historyResult = await getMatchHistory(requestedTeam1, requestedTeam2);
        let matches = [];
        if (historyResult && Array.isArray(historyResult.results)) { matches = historyResult.results; }
        else if (historyResult && Array.isArray(historyResult)) { matches = historyResult; }
        console.log("App.jsx: Raw matches received from API for history:", JSON.stringify(matches, null, 2));

        if (matches.length === 0) {
           addMessage(SENDER_BOT, 'text', { text: `No match history found containing both ${requestedTeam1} and ${requestedTeam2}.` });
        } else {
            let team1Wins = 0;
            let team2Wins = 0;
            let drawsOrNR = 0;
            let actualMatchesPlayedBetweenRequested = 0;
            // Note: Using requestedTeam1/2 directly for comparison now
            matches.forEach(match => {
                const matchTeam1Name = match.team1?.name;
                const matchTeam2Name = match.team2?.name;
                const winnerName = match.winner?.name;
                const involvesRequestedTeams =
                    (matchTeam1Name === requestedTeam1 && matchTeam2Name === requestedTeam2) ||
                    (matchTeam1Name === requestedTeam2 && matchTeam2Name === requestedTeam1);

                console.log(`Processing Match ID: ${match.id}, T1: ${matchTeam1Name}, T2: ${matchTeam2Name}, Winner: ${winnerName}, InvolvesRequested? ${involvesRequestedTeams}`);
                if (involvesRequestedTeams) {
                    actualMatchesPlayedBetweenRequested++;
                    if (!winnerName && match.result !== 'tie') { drawsOrNR++; }
                    else if (winnerName === requestedTeam1) { team1Wins++; }
                    else if (winnerName === requestedTeam2) { team2Wins++; }
                    else { drawsOrNR++; }
                } else { console.warn("Match result filtered out as teams didn't match requested H2H query:", match); }
            });
            const totalPlayed = actualMatchesPlayed;
            console.log(`App.jsx: Finished processing. Total Played (filtered H2H): ${actualMatchesPlayedBetweenRequested}, ${requestedTeam1} Wins: ${team1Wins}, ${requestedTeam2} Wins: ${team2Wins}, Draw/NR: ${drawsOrNR}`);

            if (totalPlayed === 0) {
                 addMessage(SENDER_BOT, 'text', { text: `Found related matches, but none directly between ${requestedTeam1} and ${requestedTeam2}.` });
            } else {
                addMessage(SENDER_BOT, 'historySummary', {
                  team1Name: requestedTeam1, team2Name: requestedTeam2, played: totalPlayed,
                  team1Wins: team1Wins, team2Wins: team2Wins, drawsOrNR: drawsOrNR,
                });
            }
        }
      } catch (histError) {
        console.error("App.jsx: Error caught during history fetch:", histError);
        const errorMessage = histError.response?.data ? JSON.stringify(histError.response.data) : histError.message;
        addMessage(SENDER_BOT, 'error', { text: `Failed to get match history: ${errorMessage}` });
      } finally {
        setIsCommandLoading(false);
      }
      // --- End Match History Command Logic ---

    } else {
      // --- LLM Contextual Follow-up Query Logic ---
      if (lastPredictionInfo && lastPredictionInfo.context && lastPredictionInfo.prediction) {
         addMessage(SENDER_BOT, 'text', { text: "Thinking about that..." });
         try {
            const queryPayload = {
                user_question: commandText, // Send original casing question
                match_context: lastPredictionInfo.context, // Use the stored context
                original_explanation: lastPredictionInfo.prediction.explanation || "N/A",
                predicted_winner: lastPredictionInfo.prediction.winner
            };
            console.log("App.jsx: Sending payload to LLM query endpoint:", queryPayload);
            const llmAnswer = await queryLLMContextual(queryPayload); // Call API
            if (llmAnswer && llmAnswer.answer) {
                addMessage(SENDER_BOT, 'text', { text: llmAnswer.answer });
            } else {
                 addMessage(SENDER_BOT, 'error', { text: "Sorry, I received an unexpected response when asking about that." });
            }
         } catch(llmError) {
              console.error("App.jsx: Error caught during LLM query:", llmError);
              const errorMessage = llmError.response?.data ? JSON.stringify(llmError.response.data) : llmError.message;
              addMessage(SENDER_BOT, 'error', { text: `Sorry, I couldn't process that follow-up question. Error: ${errorMessage}` });
         } finally {
              setIsCommandLoading(false);
         }
      } else {
        // No previous prediction context available, or command is not recognized.
        addMessage(SENDER_BOT, 'text', { text: "Sorry, I didn't understand that. Try 'help', or run a prediction first to ask questions about it." });
        setIsCommandLoading(false);
      }
      // --- End LLM Contextual Follow-up Query Logic ---
    }
  }; // End of handleTextInputSend

  // --- Render ---
  return (
    <div className="App bento-layout">
      <div className="bento-box input-controls">
        <div className="control-section">
          <h2>Match Prediction</h2>
          <PredictionInputForm
            formData={formData}
            handleInputChange={handleInputChange} // Handles form field changes
            onSubmitPrediction={handlePredict}    // Handles form submission for prediction
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
    </div> // End of App bento-layout
  );
}

export default App;
