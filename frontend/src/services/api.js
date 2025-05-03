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

export default apiClient; // Export if needed elsewhere
