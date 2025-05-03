// src/components/ChatMessageList.jsx
import React, { useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import Message from './Message'; // We will create this component next

function ChatMessageList({ messages, isLoading }) {
  // Create a ref to attach to the bottom of the message list
  const endOfMessagesRef = useRef(null);

  // Function to scroll to the bottom
  const scrollToBottom = () => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Use useEffect to scroll down whenever the messages array changes
  useEffect(() => {
    scrollToBottom();
  }, [messages]); // Dependency array ensures this runs when messages update

  return (
    // Add a class for styling the list container
    <div className="message-list">
      {/* Map over the messages array and render a Message component for each */}
      {messages.map((msg) => (
        <Message key={msg.id} message={msg} />
      ))}

      {/* Conditionally render a 'loading' message bubble if the bot is thinking */}
      {isLoading && (
         <Message
            // Assign a temporary key for React list rendering
            key="loading-indicator"
            message={{
                id: 'loading',
                sender: 'bot', // Displayed as a bot message
                type: 'loading',
                content: {} // No specific content needed for loading type
            }}
         />
      )}

      {/* Empty div at the end of the list. The ref is attached here. */}
      <div ref={endOfMessagesRef} />
    </div>
  );
}

// Define expected prop types
ChatMessageList.propTypes = {
  // Expect an array of message objects
  messages: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    sender: PropTypes.string.isRequired, // 'user' or 'bot'
    type: PropTypes.string.isRequired,   // 'text', 'prediction', 'error', 'loading' etc.
    content: PropTypes.object.isRequired,// Contains the actual data for the message type
  })).isRequired,
  // Expect a boolean for loading state
  isLoading: PropTypes.bool.isRequired,
};

export default ChatMessageList;
