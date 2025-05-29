import React, { useState } from 'react';
import './styles.css';

function App() {
  // State variables for inputs and predictions
  const [year, setYear] = useState('');
  const [employmentRate, setEmploymentRate] = useState('');
  const [inflationRate, setInflationRate] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null); // State to handle errors

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload
    setError(null); // Reset error state

    if (!year || !employmentRate || !inflationRate) {
      setError('All fields are required. Please fill out all inputs.');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          year: parseInt(year),
          employment_rate: parseFloat(employmentRate),
          inflation_rate: parseFloat(inflationRate),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction. Please check your backend server.');
      }

      const data = await response.json();
      setPrediction(data.gdp_prediction); // Update prediction
    } catch (err) {
      setError(err.message); // Set error message
    }
  };

  return (
    <div className="App">
      <h1>GDP Prediction</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Enter Year:</label>
          <input
            type="number"
            value={year}
            placeholder="e.g., 2025"
            onChange={(e) => setYear(e.target.value)}
          />
        </div>
        <div>
          <label>Enter Employment Rate (%):</label>
          <input
            type="number"
            step="0.01"
            value={employmentRate}
            placeholder="e.g., 5.6"
            onChange={(e) => setEmploymentRate(e.target.value)}
          />
        </div>
        <div>
          <label>Enter Inflation Rate (%):</label>
          <input
            type="number"
            step="0.01"
            value={inflationRate}
            placeholder="e.g., 2.1"
            onChange={(e) => setInflationRate(e.target.value)}
          />
        </div>
        <button type="submit">Predict</button>
      </form>

      {/* Display prediction result */}
      {prediction && (
        <div className="result">
          <h2>Predicted GDP:</h2>
          <p>{prediction.toFixed(2)} (in USD)</p>
        </div>
      )}

      {/* Display error messages */}
      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;
