// src/components/PlayerPerformance.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { getPlayerRecentPerformance, getPlayers } from '../services/api';

// --- Chart.js Imports ---
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';

// --- Register Chart.js components (Required) ---
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function PlayerPerformance() {
  const [selectedPlayerId, setSelectedPlayerId] = useState('');
  const [playerList, setPlayerList] = useState([]);
  const [isListLoading, setIsListLoading] = useState(false);
  const [listError, setListError] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [isPerfLoading, setIsPerfLoading] = useState(false);
  const [perfError, setPerfError] = useState(null);

  // Fetch Player List on Component Mount
  useEffect(() => {
    const fetchPlayers = async () => {
      setIsListLoading(true);
      setListError(null);
      try {
        // 1. Get the raw response data object from the API
        const responseData = await getPlayers();

        // 2. Extract the actual array of players
        let playerArray = [];
        if (responseData && Array.isArray(responseData.results)) {
          // Common case: Paginated response with a 'results' key
          playerArray = responseData.results;
          // Optional: Log if pagination is active (we're only getting the first page here)
          if (responseData.next) {
            console.warn("Player list is paginated, only fetching first page for dropdown.");
          }
        } else if (Array.isArray(responseData)) {
          // Less common: API directly returned an array
           playerArray = responseData;
        } else {
          // Handle unexpected structure
          console.error("Unexpected player list data structure:", responseData);
          throw new Error("Invalid data format received for player list.");
        }

        // 3. Now sort the extracted array
        playerArray.sort((a, b) => {
            // Basic sort, handle potential missing names gracefully
            const nameA = a.name || '';
            const nameB = b.name || '';
            return nameA.localeCompare(nameB);
        });

        // 4. Set state with the actual array
        setPlayerList(playerArray);

      } catch (err) {
        console.error("Failed to load player list:", err);
        // Keep the specific error message based on where the failure occurred
        setListError(err.message || "Could not load player list. Check console.");
      } finally {
        setIsListLoading(false);
      }
    };

    fetchPlayers();
  }, []); // Empty dependency array means run once on mount

  // Function to fetch performance for a specific player ID
  const fetchPerformanceForPlayer = async (playerIdToFetch) => {
    if (!playerIdToFetch) {
      setPerformanceData(null);
      setPerfError(null);
      return;
    }
    setIsPerfLoading(true);
    setPerfError(null);
    setPerformanceData(null);
    console.log(`Workspaceing performance for Player ID: ${playerIdToFetch}`);
    try {
      const data = await getPlayerRecentPerformance(playerIdToFetch);
      console.log("Received performance data:", data);
      setPerformanceData(data);
      if (data.length === 0) {
        setPerfError(`No recent performance data found for Player ID: ${playerIdToFetch}`);
      }
    } catch (err) {
      console.error("Failed to fetch player performance:", err);
      const backendError = err.response?.data ? JSON.stringify(err.response.data) : err.message;
      if (err.response?.status === 404) {
        setPerfError(`Player with ID ${playerIdToFetch} not found or has no recent performance.`);
      } else {
        setPerfError(`Failed to fetch performance: ${backendError}`);
      }
    } finally {
      setIsPerfLoading(false);
    }
  };

  // Handler for Dropdown Change
  const handlePlayerSelectionChange = (event) => {
    const newPlayerId = event.target.value;
    setSelectedPlayerId(newPlayerId);
    fetchPerformanceForPlayer(newPlayerId);
  };

  // Process data for charts using useMemo
  const chartData = useMemo(() => {
    if (!performanceData || performanceData.length === 0) return null;
    const reversedData = [...performanceData].reverse();
    const labels = reversedData.map(perf => {
      try {
        const date = new Date(perf.match_date);
        return date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
      } catch (e) { return perf.match_date; }
    });
    const runsScored = reversedData.map(perf => perf.runs_scored ?? 0);
    const strikeRate = reversedData.map(perf => perf.strike_rate ?? null);
    const wicketsTaken = reversedData.map(perf => perf.wickets_taken ?? 0);
    const economyRate = reversedData.map(perf => {
      const economy = perf.economy_rate;
      return (typeof economy === 'number' && isFinite(economy)) ? economy : null;
    });
    const didBat = runsScored.some(r => r > 0) || strikeRate.some(sr => sr !== null);
    const didBowl = wicketsTaken.some(w => w > 0) || economyRate.some(er => er !== null);
    return { labels, runsScored, strikeRate, wicketsTaken, economyRate, didBat, didBowl };
  }, [performanceData]);

  // --- Chart Configuration Options ---
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { position: 'top' }, title: { display: true }, tooltip: { mode: 'index', intersect: false } },
    scales: { x: { title: { display: true, text: 'Match Date' } } }
  };
  const battingChartOptions = {
    ...commonOptions,
    plugins: { ...commonOptions.plugins, title: { ...commonOptions.plugins.title, text: 'Batting Performance (Last 5 Matches)' } },
    scales: {
      x: { ...commonOptions.scales.x },
      yRuns: { type: 'linear', display: true, position: 'left', beginAtZero: true, title: { display: true, text: 'Runs Scored' } },
      yStrikeRate: { type: 'linear', display: true, position: 'right', beginAtZero: true, title: { display: true, text: 'Strike Rate' }, grid: { drawOnChartArea: false } }
    }
  };
  const bowlingChartOptions = {
    ...commonOptions,
    plugins: { ...commonOptions.plugins, title: { ...commonOptions.plugins.title, text: 'Bowling Performance (Last 5 Matches)' } },
    scales: {
      x: { ...commonOptions.scales.x },
      yWickets: { type: 'linear', display: true, position: 'left', beginAtZero: true, title: { display: true, text: 'Wickets Taken' }, ticks: { stepSize: 1 } },
      yEconomy: { type: 'linear', display: true, position: 'right', beginAtZero: true, title: { display: true, text: 'Economy Rate' }, grid: { drawOnChartArea: false } }
    }
  };

  // --- Chart Data Structures ---
  const battingChartData = chartData ? {
    labels: chartData.labels,
    datasets: [
      { label: 'Runs Scored', data: chartData.runsScored, backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', type: 'bar', yAxisID: 'yRuns', order: 2 },
      { label: 'Strike Rate', data: chartData.strikeRate, borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.2)', tension: 0.1, type: 'line', yAxisID: 'yStrikeRate', order: 1, spanGaps: true }
    ],
  } : null;
  const bowlingChartData = chartData ? {
    labels: chartData.labels,
    datasets: [
      { label: 'Wickets Taken', data: chartData.wicketsTaken, backgroundColor: 'rgba(75, 192, 192, 0.6)', borderColor: 'rgba(75, 192, 192, 1)', type: 'bar', yAxisID: 'yWickets', order: 2 },
      { label: 'Economy Rate', data: chartData.economyRate, borderColor: 'rgb(255, 159, 64)', backgroundColor: 'rgba(255, 159, 64, 0.2)', tension: 0.1, type: 'line', yAxisID: 'yEconomy', order: 1, spanGaps: true }
    ],
  } : null;

  return (
    <div className="player-performance-container">
      <h2>Player Recent Performance (Last 5 Matches)</h2>

      <div className="player-selection">
        <label htmlFor="playerSelect">Select Player:</label>
        <select
          id="playerSelect"
          value={selectedPlayerId}
          onChange={handlePlayerSelectionChange}
          disabled={isListLoading}
        >
          {isListLoading && <option value="">Loading players...</option>}
          {listError && <option value="">Error loading players</option>}
          {!isListLoading && !listError && (
            <option value="">-- Select a Player --</option>
          )}
          {!isListLoading && !listError && playerList.map((player) => (
            <option key={player.id} value={player.id}>
              {player.name}
            </option>
          ))}
        </select>
        {listError && <p className="error-message" style={{marginLeft: '10px'}}>{listError}</p>}
      </div>

      <div className="performance-display">
        {isPerfLoading && <p>Loading performance data...</p>}
        {perfError && <p className="error-message">Error: {perfError}</p>}

        {/* Render Charts */}
        {!isPerfLoading && !perfError && chartData && (
          <div>
            {chartData.didBat && battingChartData && (
              <div className="chart-container">
                <Bar data={battingChartData} options={battingChartOptions} />
              </div>
            )}
            {!isPerfLoading && !perfError && chartData && !chartData.didBat && selectedPlayerId && <p>No recent batting data available for the selected player.</p>}

            {chartData.didBowl && bowlingChartData && (
              <div className="chart-container">
                 <Line data={bowlingChartData} options={bowlingChartOptions} />
              </div>
            )}
             {!isPerfLoading && !perfError && chartData && !chartData.didBowl && selectedPlayerId && <p>No recent bowling data available for the selected player.</p>}
          </div>
        )}

        {/* Initial message */}
        {!isPerfLoading && !perfError && !performanceData && !selectedPlayerId && (
             <p>Select a player from the dropdown to see recent trends.</p>
        )}
      </div>
    </div>
  );
}

export default PlayerPerformance;
