// src/services/api.js
import axios from "axios";

// Get the base URL from environment variables, defaulting to your Django dev server
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

// Function to get the auth token (we'll store it securely later)
const getAuthToken = () => {
  // For now, let's hardcode it for testing, but REMOVE THIS LATER
  // return 'YOUR_DJANGO_API_TOKEN'; // Replace with a real token for testing
  // Better: Read from localStorage or secure storage after login
  return localStorage.getItem("authToken");
};

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 second timeout
  headers: {
    "Content-Type": "application/json",
  },
});

// Add a request interceptor to include the token dynamically
apiClient.interceptors.request.use(
  (config) => {
    const token = getAuthToken();
    if (token) {
      config.headers["Authorization"] = `Token ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export const predictMatch = async (matchData) => {
  try {
    // Adjust the endpoint path if needed, assuming '/predict/' is relative to API_BASE_URL
    const response = await apiClient.post("/predict/", matchData);
    return response.data;
  } catch (error) {
    console.error(
      "Error fetching prediction:",
      error.response || error.message
    );
    // Rethrow or handle error appropriately
    throw error;
  }
};

export const getPlayerRecentPerformance = async (playerId) => {
  if (!playerId) {
    throw new Error("Player ID is required");
  }
  try {
    // Use template literal to insert player ID into the URL
    const response = await apiClient.get(
      `/players/${playerId}/recent-performance/`
    );
    return response.data; // The API returns a list of performance objects
  } catch (error) {
    console.error(
      `Error fetching recent performance for player ${playerId}:`,
      error.response?.data || error.message
    );
    throw error; // Re-throw to be caught by the component
  }
};

export const getPlayers = async (queryParams = '') => {
  const queryString = queryParams && typeof queryParams === 'string' ? (queryParams.startsWith('?') ? queryParams : `?${queryParams}`) : '';
  const url = `/players/${queryString}`;

  console.log("api.js: getPlayers requesting URL:", url);

  try {
    const response = await apiClient.get(url);
    console.log("api.js: getPlayers returning response.data:", JSON.stringify(response.data, null, 2));
    return response.data;
  } catch (error) {
    // Log specific error for this function
    console.error(`Error fetching players with url '${url}':`, error.response?.data || error.message);
    throw error;
  }
};

export const predictScore = async (matchData) => {
  try {
    // Assuming the Django endpoint is /api/v1/predict_score/
    const response = await apiClient.post('/predict_score/', matchData);
    // Adjust response field based on actual API (e.g., predicted_first_innings_score)
    return response.data; // Example: expects { predicted_score: 175.5 }
  } catch (error) {
    console.error("Error fetching score prediction:", error.response?.data || error.message);
    throw error;
  }
};

export const getMatchHistory = async (team1Name, team2Name) => {
  // Basic validation
  if (!team1Name || !team2Name) {
    throw new Error("Two team names are required for history lookup.");
  }
  try {
    const queryParams = `?search=${encodeURIComponent(team1Name)}&search=${encodeURIComponent(team2Name)}&ordering=-date&page_size=200`;
    const url = `/matches/${queryParams}`;

    console.log("api.js: getMatchHistory requesting URL:", url);
    const response = await apiClient.get(url);
    return response.data; // Expects direct array or { count: ..., results: [...] }
  } catch (error) {
    console.error(`Error fetching match history for ${team1Name} vs ${team2Name}:`, error.response?.data || error.message);
    throw error;
  }
};




export default apiClient; // Export if needed elsewhere
