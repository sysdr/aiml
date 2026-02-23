import React from 'react';
import { ThemeProvider } from 'styled-components';
import { GlobalStyles, theme } from './styles/GlobalStyles';
import ChatContainer from './components/ChatContainer';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles />
      <div className="App">
        <ChatContainer />
      </div>
    </ThemeProvider>
  );
}

export default App;
