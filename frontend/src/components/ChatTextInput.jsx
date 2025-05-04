// src/components/ChatTextInput.jsx
import React, { useState } from 'react';
import PropTypes from 'prop-types';

function ChatTextInput({ onSendMessage, isLoading }) {
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSend = (event) => {
    // Prevent default form submission if wrapped in a form tag later
    event.preventDefault();
    const trimmedInput = inputValue.trim();
    if (trimmedInput) {
      onSendMessage(trimmedInput); // Send the trimmed message up to App.jsx
      setInputValue(''); // Clear the input field
    }
  };

  // Allow sending message by pressing Enter key
  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter, allow Shift+Enter for newline (if needed)
      handleSend(event);
    }
  };

  return (
    // Use a form tag to allow Enter key submission
    <form onSubmit={handleSend} className="chat-text-input-form">
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyPress={handleKeyPress} // Add keypress handler
        placeholder="Type 'help' or ask for player performance..."
        disabled={isLoading} // Disable input if parent is loading
        aria-label="Chat message input" // Accessibility
      />
      <button className="chat-send-button" type="submit" disabled={isLoading || !inputValue.trim()}>
        Send
      </button>
    </form>
  );
}

ChatTextInput.propTypes = {
  onSendMessage: PropTypes.func.isRequired, // Callback to send message to parent
  isLoading: PropTypes.bool.isRequired,    // To disable input during API calls
};

export default ChatTextInput;
