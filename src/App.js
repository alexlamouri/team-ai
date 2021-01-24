import React, { useState, useEffect } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [before, setBefore] = useState();
  const [after, setAfter] = useState();

  const handleChange = (event) => {
    setQuery(event.target.value);
  };

  const scrape = () => {
    fetch("/api/scrape?query=" + query, {
      method: "GET",
    })
      .then((res) => res.json())
      .then((data) => {
        setBefore(data.before);
        setAfter(data.after);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <div>
          <p>Enter a query:</p>
          <input onChange={handleChange}></input>
          <button onClick={scrape}>Search</button>
        </div>
        <div>
          <p>Here is the impact of COVID-19 on the query:</p>
          <div>before {before}</div>
          <br></br>
          <div>after {after}</div>
        </div>
      </header>
    </div>
  );
}

export default App;
