// src/components/PlayerPerformance.jsx
import React, { useState, useMemo } from "react";
import { getPlayerRecentPerformance } from "../services/api";

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
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

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
  const [playerId, setPlayerId] = useState("");
  const [performanceData, setPerformanceData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetchPerformance = async () => {
    if (!playerId) {
      setError("Please enter a Player ID.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setPerformanceData(null);
    console.log(`Workspaceing performance for Player ID: ${playerId}`);

    try {
      const data = await getPlayerRecentPerformance(playerId);
      console.log("Received performance data:", data);
      setPerformanceData(data);
      if (data.length === 0) {
        setError(`No recent performance data found for Player ID: ${playerId}`);
      }
    } catch (err) {
      console.error("Failed to fetch player performance:", err);
      const backendError = err.response?.data
        ? JSON.stringify(err.response.data)
        : err.message;
      if (err.response?.status === 404) {
        setError(
          `Player with ID ${playerId} not found or has no recent performance.`
        );
      } else {
        setError(`Failed to fetch performance: ${backendError}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Process data for charts using useMemo for efficiency
  const chartData = useMemo(() => {
    if (!performanceData || performanceData.length === 0) {
      return null;
    }
    const reversedData = [...performanceData].reverse(); // Display oldest match first
    const labels = reversedData.map((perf) => {
      try {
        const date = new Date(perf.match_date);
        return date.toLocaleDateString("en-GB", {
          day: "2-digit",
          month: "short",
        });
      } catch (e) {
        return perf.match_date;
      }
    });

    const runsScored = reversedData.map((perf) => perf.runs_scored ?? 0);
    const strikeRate = reversedData.map((perf) => perf.strike_rate ?? null);
    const wicketsTaken = reversedData.map((perf) => perf.wickets_taken ?? 0);
    const economyRate = reversedData.map((perf) => {
      const economy = perf.economy_rate;
      return typeof economy === "number" && isFinite(economy) ? economy : null;
    });

    const didBat =
      runsScored.some((r) => r > 0) || strikeRate.some((sr) => sr !== null);
    const didBowl =
      wicketsTaken.some((w) => w > 0) || economyRate.some((er) => er !== null);

    return {
      labels,
      runsScored,
      strikeRate,
      wicketsTaken,
      economyRate,
      didBat,
      didBowl,
    };
  }, [performanceData]);

  // --- Chart Configuration Options ---
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true },
      tooltip: { mode: "index", intersect: false },
    },
    scales: { x: { title: { display: true, text: "Match Date" } } }, // Base X axis
  };

  const battingChartOptions = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        ...commonOptions.plugins.title,
        text: "Batting Performance (Last 5 Matches)",
      },
    },
    scales: {
      // Multi-axis configuration
      x: { ...commonOptions.scales.x },
      yRuns: {
        type: "linear",
        display: true,
        position: "left",
        beginAtZero: true,
        title: { display: true, text: "Runs Scored" },
      },
      yStrikeRate: {
        type: "linear",
        display: true,
        position: "right",
        beginAtZero: true,
        title: { display: true, text: "Strike Rate" },
        grid: { drawOnChartArea: false },
      },
    },
  };

  const bowlingChartOptions = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        ...commonOptions.plugins.title,
        text: "Bowling Performance (Last 5 Matches)",
      },
    },
    scales: {
      // Multi-axis configuration
      x: { ...commonOptions.scales.x },
      yWickets: {
        type: "linear",
        display: true,
        position: "left",
        beginAtZero: true,
        title: { display: true, text: "Wickets Taken" },
        ticks: { stepSize: 1 },
      },
      yEconomy: {
        type: "linear",
        display: true,
        position: "right",
        beginAtZero: true,
        title: { display: true, text: "Economy Rate" },
        grid: { drawOnChartArea: false },
      },
    },
  };

  // --- Chart Data Structures ---
  const battingChartData = chartData
    ? {
        labels: chartData.labels,
        datasets: [
          {
            label: "Runs Scored",
            data: chartData.runsScored,
            backgroundColor: "rgba(54, 162, 235, 0.6)",
            borderColor: "rgba(54, 162, 235, 1)",
            type: "bar",
            yAxisID: "yRuns",
            order: 2,
          },
          {
            label: "Strike Rate",
            data: chartData.strikeRate,
            borderColor: "rgb(255, 99, 132)",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            tension: 0.1,
            type: "line",
            yAxisID: "yStrikeRate",
            order: 1,
            spanGaps: true,
          },
        ],
      }
    : null;

  const bowlingChartData = chartData
    ? {
        labels: chartData.labels,
        datasets: [
          {
            label: "Wickets Taken",
            data: chartData.wicketsTaken,
            backgroundColor: "rgba(75, 192, 192, 0.6)",
            borderColor: "rgba(75, 192, 192, 1)",
            type: "bar",
            yAxisID: "yWickets",
            order: 2,
          },
          {
            label: "Economy Rate",
            data: chartData.economyRate,
            borderColor: "rgb(255, 159, 64)",
            backgroundColor: "rgba(255, 159, 64, 0.2)",
            tension: 0.1,
            type: "line",
            yAxisID: "yEconomy",
            order: 1,
            spanGaps: true,
          },
        ],
      }
    : null;

  return (
    <div className="player-performance-container">
      <h2>Player Recent Performance (Last 5 Matches)</h2>

      <div className="player-selection">
        <label htmlFor="playerIdInput">Enter Player ID:</label>
        <input
          type="number"
          id="playerIdInput"
          value={playerId}
          onChange={(e) => setPlayerId(e.target.value)}
          placeholder="e.g., 123"
        />
        <button onClick={handleFetchPerformance} disabled={isLoading}>
          {isLoading ? "Loading..." : "Get Performance"}
        </button>
        <p className="id-note">
          (Note: Find Player IDs via the{" "}
          <a
            href="/api/schema/swagger-ui/#/players/players_list"
            target="_blank"
            rel="noopener noreferrer"
          >
            Players API
          </a>
          )
        </p>
      </div>

      <div className="performance-display">
        {isLoading && <p>Loading performance data...</p>}
        {error && <p className="error-message">Error: {error}</p>}

        {/* Render Charts */}
        {chartData && (
          <div>
            {/* Keep the H3 or remove if chart titles are sufficient */}
            {/* <h3>Performance Trend for Player ID: {playerId}</h3> */}

            {chartData.didBat && battingChartData && (
              <div className="chart-container">
                {/* Using Bar type allows mixed chart defined in datasets */}
                <Bar data={battingChartData} options={battingChartOptions} />
              </div>
            )}
            {!isLoading && !error && chartData && !chartData.didBat && (
              <p>No recent batting data available for Player ID {playerId}.</p>
            )}

            {chartData.didBowl && bowlingChartData && (
              <div className="chart-container">
                {/* Using Line type allows mixed chart defined in datasets */}
                <Line data={bowlingChartData} options={bowlingChartOptions} />
              </div>
            )}
            {!isLoading && !error && chartData && !chartData.didBowl && (
              <p>No recent bowling data available for Player ID {playerId}.</p>
            )}
          </div>
        )}
        {/* Initial message */}
        {!isLoading && !error && !performanceData && (
          <p>
            Enter a Player ID and click "Get Performance" to see recent trends.
          </p>
        )}
        {/* Message if data fetched but was empty list */}
        {!isLoading &&
          !error &&
          performanceData &&
          performanceData.length === 0 && (
            <p>No recent performance data found for Player ID {playerId}.</p>
          )}
      </div>
    </div>
  );
}

export default PlayerPerformance;
