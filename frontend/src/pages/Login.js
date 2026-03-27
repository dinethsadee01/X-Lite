import React, { useState } from 'react';
import { Box, Card, CardContent, Typography, TextField, Button, Alert } from '@mui/material';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await login(username, password);
            navigate('/');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to login. Please check credentials.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={{ 
            height: '100vh', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            background: 'radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.20), rgba(244, 249, 255, 0.95) 45%), #f4f9ff'
        }}>
            <Card sx={{ maxWidth: 440, width: '100%', p: 2.5, boxShadow: '0 20px 45px rgba(29, 78, 216, 0.12)' }}>
                <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
                        <MonitorHeartIcon sx={{ color: '#1d4ed8', fontSize: 52, mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 800, color: '#1e3a8a' }}>
                            X-Lite Medical
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                            Radiological AI Analysis Portal
                        </Typography>
                    </Box>

                    {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

                    <form onSubmit={handleSubmit}>
                        <TextField
                            fullWidth
                            label="Username"
                            variant="outlined"
                            margin="normal"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                        <TextField
                            fullWidth
                            label="Password"
                            type="password"
                            variant="outlined"
                            margin="normal"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            size="large"
                            disabled={loading}
                            sx={{
                                mt: 3,
                                bgcolor: '#1d4ed8',
                                '&:hover': { bgcolor: '#1e40af' },
                                py: 1.5
                            }}
                        >
                            {loading ? 'Authenticating...' : 'Sign In'}
                        </Button>
                    </form>
                    
                    <Typography variant="caption" display="block" textAlign="center" sx={{ mt: 3, color: 'text.secondary' }}>
                        Authorized Clinical Personnel Only
                    </Typography>
                </CardContent>
            </Card>
        </Box>
    );
}
