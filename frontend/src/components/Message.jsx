// src/components/Message.jsx
import React from 'react';
import PropTypes from 'prop-types';
import ConfidenceBar from './ConfidenceBar'; // Reuse the confidence bar component

function Message({ message }) {
  // Destructure properties from the message object
  const { sender, type, content } = message;
  // Determine if the sender is the user for styling purposes
  const isUser = sender === 'user';

  // Helper function to render the correct content based on message type
  const renderContent = () => {
    switch (type) {
      case 'text':
        // Simple text message
        return <p>{content.text}</p>;

      case 'prediction':
        // Structured message for prediction results
        return (
          <div>
            {/* Display predicted winner */}
            <p><strong>Prediction:</strong> {content.winner || 'N/A'}</p>

            {/* Display confidence bar if score exists */}
            {typeof content.confidence === 'number' ? (
              <div style={{ margin: '8px 0' }}> {/* Add some spacing */}
                <ConfidenceBar score={content.confidence} />
              </div>
            ) : <p><i>Confidence: N/A</i></p>}

            {/* Display explanation if it exists */}
            {content.explanation && content.explanation.trim() !== '' ? (
              <>
                <p style={{ marginTop: '5px', marginBottom: '5px' }}><strong>Explanation:</strong></p>
                {/* Add specific class for styling within chat */}
                <pre className="explanation-box chat-explanation">
                  {content.explanation}
                </pre>
              </>
            ) : (
              <p><i>No explanation provided.</i></p>
            )}
          </div>
        );

      case 'error':
        // Error message from the bot
        return <p className="error-text">⚠️ {content.text}</p>; // Add class for styling

      case 'loading':
        // Loading indicator (animated dots from CSS)
        return <p className="loading-dots"><span>.</span><span>.</span><span>.</span></p>;

      default:
        // Fallback for unknown message types
        return <p><i>Unsupported message format.</i></p>;
    }
  };

  // Render the message bubble with appropriate classes
  return (
    <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
      {renderContent()}
    </div>
  );
}

// Define expected prop types for the message object
Message.propTypes = {
  message: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    sender: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    content: PropTypes.object.isRequired, // Content structure varies by type
  }).isRequired,
};

export default Message;
