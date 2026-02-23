import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AI Chat Assistant', () => {
  render(<App />);
  const titleElement = screen.getByText(/AI Chat Assistant/i);
  expect(titleElement).toBeInTheDocument();
});
