import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { Send, Bot, User, Loader, AlertCircle } from 'lucide-react';
import { chatAPI } from '../services/api';

const Container = styled.div`
  width: 100%;
  max-width: 800px;
  height: 600px;
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius['2xl']};
  box-shadow: ${props => props.theme.shadows.xl};
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const Header = styled.div`
  background: ${props => props.theme.colors.primary};
  color: white;
  padding: 1.5rem;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
`;

const Subtitle = styled.p`
  font-size: 0.875rem;
  opacity: 0.9;
  margin: 0.5rem 0 0 0;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const Message = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  ${props => props.$isUser && 'flex-direction: row-reverse;'}
`;

const MessageIcon = styled.div`
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${props => props.$isUser ? props.theme.colors.primary : props.theme.colors.secondary};
  color: white;
`;

const MessageBubble = styled.div`
  max-width: 70%;
  padding: 0.75rem 1rem;
  border-radius: ${props => props.theme.borderRadius.lg};
  background: ${props => props.$isUser ? props.theme.colors.primary : props.theme.colors.surface};
  color: ${props => props.$isUser ? 'white' : props.theme.colors.text};
  word-wrap: break-word;
  line-height: 1.5;
`;

const InputContainer = styled.div`
  padding: 1rem;
  border-top: 1px solid ${props => props.theme.colors.border};
  background: ${props => props.theme.colors.surface};
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
`;

const TextArea = styled.textarea`
  flex: 1;
  min-height: 2.5rem;
  max-height: 6rem;
  padding: 0.75rem;
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  font-family: inherit;
  font-size: 0.875rem;
  resize: vertical;
  transition: border-color 0.2s;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }

  &:disabled {
    background: ${props => props.theme.colors.border};
    cursor: not-allowed;
  }
`;

const SendButton = styled.button`
  width: 2.5rem;
  height: 2.5rem;
  border: none;
  border-radius: ${props => props.theme.borderRadius.lg};
  background: ${props => props.theme.colors.primary};
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover:not(:disabled) {
    background: ${props => props.theme.colors.primaryLight};
    transform: translateY(-1px);
  }

  &:disabled {
    background: ${props => props.theme.colors.border};
    cursor: not-allowed;
    transform: none;
  }
`;

const LoadingMessage = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${props => props.theme.colors.textLight};
  font-style: italic;
`;

const ErrorMessage = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${props => props.theme.colors.error};
  background: ${props => props.theme.colors.error}10;
  padding: 0.75rem;
  border-radius: ${props => props.theme.borderRadius.md};
  margin: 0.5rem 0;
`;

const ChatContainer = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant. I\'m here to help you learn Python and AI. What would you like to know?',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    const el = messagesEndRef.current;
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError('');

    try {
      const response = await chatAPI.sendMessage(userMessage.content, messages);
      
      if (response.success) {
        const assistantMessage = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(response.error_message || 'Failed to get response');
      }
    } catch (err) {
      setError(err.message);
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Container>
      <Header>
        <Title>AI Chat Assistant</Title>
        <Subtitle>Powered by Python & Gemini AI - Day 1 Project</Subtitle>
      </Header>

      <MessagesContainer>
        {messages.map((message, index) => (
          <Message key={index} $isUser={message.role === 'user'}>
            <MessageIcon $isUser={message.role === 'user'}>
              {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
            </MessageIcon>
            <MessageBubble $isUser={message.role === 'user'}>
              {message.content}
            </MessageBubble>
          </Message>
        ))}

        {isLoading && (
          <LoadingMessage>
            <Loader size={16} className="animate-spin" />
            AI is thinking...
          </LoadingMessage>
        )}

        {error && (
          <ErrorMessage>
            <AlertCircle size={16} />
            {error}
          </ErrorMessage>
        )}

        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <InputWrapper>
          <TextArea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here..."
            disabled={isLoading}
            rows={1}
          />
          <SendButton
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
          >
            {isLoading ? <Loader size={16} /> : <Send size={16} />}
          </SendButton>
        </InputWrapper>
      </InputContainer>
    </Container>
  );
};

export default ChatContainer;
