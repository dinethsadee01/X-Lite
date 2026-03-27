import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';

import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Header from './components/Header';
import Login from './pages/Login';
import Home from './pages/Home';
import Instructions from './pages/Instructions';
import Dashboard from './pages/Dashboard';
import Results from './pages/Results';
import History from './pages/History';
import './App.css';

const theme = createTheme({
    palette: {
        primary: {
            main: '#1d4ed8',
            light: '#60a5fa',
            dark: '#1e3a8a',
        },
        secondary: {
            main: '#0ea5e9',
        },
        info: {
            main: '#38bdf8',
        },
        background: {
            default: '#f4f9ff',
            paper: '#ffffff',
        },
        text: {
            primary: '#0f172a',
            secondary: '#475569',
        },
    },
    typography: {
        fontFamily: '"Manrope", "Nunito Sans", "Segoe UI", sans-serif',
        h4: {
            fontWeight: 800,
            letterSpacing: '-0.02em',
        },
        h5: {
            fontWeight: 800,
            letterSpacing: '-0.01em',
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    background:
                        'radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.18) 0%, rgba(244, 249, 255, 1) 35%), radial-gradient(circle at 90% 0%, rgba(125, 211, 252, 0.2) 0%, rgba(244, 249, 255, 0.95) 30%), #f4f9ff',
                    minHeight: '100vh',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: '12px',
                    fontWeight: 700,
                    boxShadow: 'none',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: '18px',
                    border: '1px solid #dbeafe',
                    boxShadow: '0 8px 30px rgba(30, 64, 175, 0.08)',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    borderRadius: '16px',
                },
            },
        },
    },
});

function MainLayout({ children }) {
    return (
        <Box className="app-shell">
            <Header />
            <main style={{ paddingTop: 14 }}>{children}</main>
        </Box>
    );
}

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <AuthProvider>
                <Router>
                    <Routes>
                        <Route path="/login" element={<Login />} />
                        
                        {/* Protected Routes */}
                        <Route path="/" element={
                            <ProtectedRoute>
                                <MainLayout>
                                    <Home />
                                </MainLayout>
                            </ProtectedRoute>
                        } />

                        <Route path="/instructions" element={
                            <ProtectedRoute>
                                <MainLayout>
                                    <Instructions />
                                </MainLayout>
                            </ProtectedRoute>
                        } />

                        <Route path="/upload" element={
                            <ProtectedRoute>
                                <MainLayout>
                                    <Dashboard />
                                </MainLayout>
                            </ProtectedRoute>
                        } />
                        
                        <Route path="/results" element={
                            <ProtectedRoute>
                                <MainLayout>
                                    <Results />
                                </MainLayout>
                            </ProtectedRoute>
                        } />
                        
                        <Route path="/history" element={
                            <ProtectedRoute>
                                <MainLayout>
                                    <History />
                                </MainLayout>
                            </ProtectedRoute>
                        } />

                        {/* Catch all */}
                        <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                </Router>
            </AuthProvider>
        </ThemeProvider>
    );
}

export default App;
