// src/components/Message.jsx
import React from 'react'; // useMemo might not be needed if calculating directly
import PropTypes from 'prop-types';
import ConfidenceBar from './ConfidenceBar';
// --- Import Chart Components ---
import { Bar, Line } from 'react-chartjs-2';

// Note: Chart.js elements should be registered globally (e.g., in App.jsx or main.jsx)
// If not, you'd need to import and register them here, which is less ideal.

function Message({ message }) {
  const { sender, type, content } = message;
  const isUser = sender === 'user';

  // Helper function to render the correct content based on message type
  const renderContent = () => {
    switch (type) {
      case 'text':
        return <p>{content.text}</p>;

      case 'prediction':
        // ... (prediction rendering logic remains the same as before) ...
        return (
          <div>
            {content.winner && <p><strong>Prediction:</strong> {content.winner}</p>}
            {typeof content.predicted_score === 'number' && (
               <p><strong>Predicted 1st Innings Score:</strong> {Math.round(content.predicted_score)}</p>
            )}
            {typeof content.confidence === 'number' ? (
              <div className="confidence-display" style={{ margin: '8px 0' }}>
                 <span style={{ fontWeight: 'bold', marginRight: '8px', flexShrink: 0 }}>Confidence:</span>
                 <ConfidenceBar score={content.confidence} />
              </div>
            ) : (<p><i>Confidence: N/A</i></p>)}
            {content.explanation && content.explanation.trim() !== '' ? (
                 <>
                    <p style={{ marginTop: '5px', marginBottom: '5px' }}><strong>Explanation:</strong></p>
                    <pre className="explanation-box chat-explanation">{content.explanation}</pre>
                 </>
            ) : ( content.winner && <p style={{marginTop: '5px'}}><i>No explanation provided.</i></p> )}
          </div>
        );

      // --- NEW CASE FOR PERFORMANCE CHARTS ---
      case 'performanceChart':
        // Process data directly here or use a helper function/hook if complex
        const performanceData = content.performanceData; // The array passed from App.jsx
        const playerName = content.playerName;

        if (!performanceData || performanceData.length === 0) {
          return <p>No performance data available to display chart.</p>;
        }

        // Process data similar to how it was done in PlayerPerformance component
        const reversedData = [...performanceData].reverse();
        const labels = reversedData.map(perf => new Date(perf.match_date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }));
        const runsScored = reversedData.map(perf => perf.runs_scored ?? 0);
        const strikeRate = reversedData.map(perf => perf.strike_rate ?? null);
        const wicketsTaken = reversedData.map(perf => perf.wickets_taken ?? 0);
        const economyRate = reversedData.map(perf => {
            const economy = perf.economy_rate;
            return (typeof economy === 'number' && isFinite(economy)) ? economy : null;
        });
        const didBat = runsScored.some(r => r > 0) || strikeRate.some(sr => sr !== null);
        const didBowl = wicketsTaken.some(w => w > 0) || economyRate.some(er => er !== null);

        // Define Chart Options (incorporate dark mode colors)
        // Using hardcoded colors matching CSS vars for simplicity here
        const chartTextColor = '#a0a0a8'; // --text-secondary
        const chartTitleColor = '#e1e1e1'; // --text-primary
        const commonOptions = {
            responsive: true, maintainAspectRatio: true,
            plugins: {
                legend: { position: 'top', labels: { color: chartTextColor } },
                title: { display: true, color: chartTitleColor }, // Title text set below
                tooltip: { titleColor: chartTitleColor, bodyColor: chartTitleColor }
            },
            scales: { x: { ticks: { color: chartTextColor }, title: { display: true, text: 'Match Date', color: chartTitleColor } } }
        };
        const battingChartOptions = { ...commonOptions, plugins: { ...commonOptions.plugins, title: { ...commonOptions.plugins.title, text: 'Batting Trend' } }, scales: { x: { ...commonOptions.scales.x }, yRuns: { type: 'linear', display: true, position: 'left', beginAtZero: true, title: { display: true, text: 'Runs', color: chartTitleColor }, ticks:{ color: chartTextColor } }, yStrikeRate: { type: 'linear', display: true, position: 'right', beginAtZero: true, title: { display: true, text: 'SR', color: chartTitleColor }, grid: { drawOnChartArea: false }, ticks:{ color: chartTextColor } } } };
        const bowlingChartOptions = { ...commonOptions, plugins: { ...commonOptions.plugins, title: { ...commonOptions.plugins.title, text: 'Bowling Trend' } }, scales: { x: { ...commonOptions.scales.x }, yWickets: { type: 'linear', display: true, position: 'left', beginAtZero: true, title: { display: true, text: 'Wkts', color: chartTitleColor }, ticks: { stepSize: 1, color: chartTextColor } }, yEconomy: { type: 'linear', display: true, position: 'right', beginAtZero: true, title: { display: true, text: 'Econ', color: chartTitleColor }, grid: { drawOnChartArea: false }, ticks:{ color: chartTextColor } } } };

        // Define Chart Data
        const battingChartData = { labels, datasets: [ { label: 'Runs', data: runsScored, backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', type: 'bar', yAxisID: 'yRuns', order: 2 }, { label: 'SR', data: strikeRate, borderColor: 'rgb(255, 99, 132)', tension: 0.1, type: 'line', yAxisID: 'yStrikeRate', order: 1, spanGaps: true } ] };
        const bowlingChartData = { labels, datasets: [ { label: 'Wkts', data: wicketsTaken, backgroundColor: 'rgba(75, 192, 192, 0.6)', borderColor: 'rgba(75, 192, 192, 1)', type: 'bar', yAxisID: 'yWickets', order: 2 }, { label: 'Econ', data: economyRate, borderColor: 'rgb(255, 159, 64)', tension: 0.1, type: 'line', yAxisID: 'yEconomy', order: 1, spanGaps: true } ] };

        return (
          <div>
            <p style={{marginBottom: '8px'}}><strong>Recent Performance for {playerName}:</strong></p>
            {didBat && (
              <div className="chart-container chat-chart"> {/* Add specific class */}
                <Bar data={battingChartData} options={battingChartOptions} />
              </div>
            )}
            {didBowl && (
              <div className="chart-container chat-chart"> {/* Add specific class */}
                 <Line data={bowlingChartData} options={bowlingChartOptions} />
              </div>
            )}
            {!didBat && !didBowl && <p><i>No specific stats found in recent matches.</i></p>}
          </div>
        );
      // --- END NEW CASE ---

      case 'historySummary':
        return (
          <div className="history-summary"> {/* Add class for potential styling */}
            <p><strong>Head-to-Head: {content.team1Name} vs {content.team2Name}</strong></p>
            {/* Optional separator */}
            <hr className="summary-divider" />
            {/* Use a list or paragraphs for stats */}
            <ul>
                <li>Played: {content.played}</li>
                <li>{content.team1Name} Won: {content.team1Wins}</li>
                <li>{content.team2Name} Won: {content.team2Wins}</li>
                <li>Draw/No Result: {content.drawsOrNR}</li>
            </ul>
          </div>
        );



      case 'error':
        return <p className="error-text">⚠️ {content.text}</p>;
      case 'loading':
        return <p className="loading-dots"><span>.</span><span>.</span><span>.</span></p>;
      default:
        return <p><i>Unsupported message format.</i></p>;
    }
  };

  return (
    <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
      {renderContent()}
    </div>
  );
}

// Update PropTypes to include new content structure for performanceChart
Message.propTypes = {
  message: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    sender: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    content: PropTypes.shape({
        // Existing types
        text: PropTypes.string,
        winner: PropTypes.string,
        confidence: PropTypes.number,
        explanation: PropTypes.string,
        predicted_score: PropTypes.number,
        playerName: PropTypes.string,
        performanceData: PropTypes.array,
        // New type content for history
        team1Name: PropTypes.string,
        team2Name: PropTypes.string,
        played: PropTypes.number,
        team1Wins: PropTypes.number,
        team2Wins: PropTypes.number,
        drawsOrNR: PropTypes.number,
    }).isRequired,
  }).isRequired,
};

export default Message;
