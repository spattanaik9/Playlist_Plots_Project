/*import React, { useState } from 'react';

function App() {
  const [title, setTitle] = useState('');
  const [author, setAuthor] = useState('');
  const [results, setResults] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Fetch data from the Open Library API based on user input
    const response = await fetch(`https://openlibrary.org/search.json?q=${title}+${author}`);
    const data = await response.json();
    // Update state with the results
    setResults(data.docs);
  };

  return (
    <div className="App">
      <h1>Book Search</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Book Title:
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>
        <br />
        <label>
          Author Name:
          <input
            type="text"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
          />
        </label>
        <br />
        <button type="submit">Search</button>
      </form>
      <div>
        {results.length > 0 ? (
          <ul>
            {results.map((book, index) => (
              <li key={index}>
                <strong>Title:</strong> {book.title}<br />
                {book.author_name ? <><strong>Author:</strong> {book.author_name.join(', ')}<br /></> : ''}
                {book.first_publish_year ? <><strong>First Published:</strong> {book.first_publish_year}<br /></> : ''}
            
                {book.key}
              </li>
            ))}
          </ul>
        ) : (
          <p>No results found.</p>
        )}
      </div>
    </div>
  );
}

export default App;*/