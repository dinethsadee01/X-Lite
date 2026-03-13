import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';

import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Header from './components/Header';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Results from './pages/Results';
import History from './pages/History';
import './App.css';

const theme = createTheme({
    palette: {
        primary: {
            main: '#1E3A8A', // Deep Blue for medical trust
        },
        secondary: {
            main: '#0284c7', // Lighter blue
        },
        background: {
            default: '#f8fafc',
        }
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: '8px',
                    fontWeight: 600,
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: '12px',
                },
            },
        },
    },
});

function MainLayout({ children }) {
    return (
        <>
            <Header />
            <main>{children}</main>
        </>
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
