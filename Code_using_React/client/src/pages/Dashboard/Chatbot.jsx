import React, { useState } from 'react';
import axios from 'axios';
import './Chatbot.css';  // Optional: CSS for styling

const Chatbot = () => {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  // Handler for user input
  const handleUserInput = async () => {
    if (!userInput.trim()) return;

    const newMessage = { sender: 'user', text: userInput };
    setChatHistory((prevHistory) => [...prevHistory, newMessage]);

    try {
      // Call the FastAPI ask endpoint
      const response = await axios.post('http://localhost:8000/ask', {
        user_question: userInput,
      });

      // Append the response to the chat history
      const botMessage = { sender: 'bot', text: response.data };
      setChatHistory((prevHistory) => [...prevHistory, botMessage]);
    } catch (error) {
      console.error('Error fetching bot response:', error);
      const errorMessage = { sender: 'bot', text: 'Error processing your request.' };
      setChatHistory((prevHistory) => [...prevHistory, errorMessage]);
    }

    // Clear the input field
    setUserInput('');
  };

  return (
    <div className="chatbot-container">
      <div className="chat-window">
        {chatHistory.map((message, index) => (
          <div key={index} className={`chat-message ${message.sender}`}>
            {message.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button onClick={handleUserInput}>Send</button>
      </div>
    </div>
  );
};

export default Chatbot;
