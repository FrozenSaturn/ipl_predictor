// src/App.jsx
import React from "react";
import PredictionForm from "./components/PredictionForm";
import PlayerPerformance from "./components/PlayerPerformance";
import "./App.css"; // Or remove if you don't use it

function App() {
  return (
    <div className="App">
      <PredictionForm />
      <PlayerPerformance />
    </div>
  );
}

export default App;
