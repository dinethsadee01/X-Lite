import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Avatar } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';

export default function Header() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <AppBar position="static" color="transparent" elevation={1} sx={{ backgroundColor: 'white', mb: 4 }}>
            <Toolbar>
                <MonitorHeartIcon sx={{ color: '#1E3A8A', mr: 1, fontSize: 32 }} />
                <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold', color: '#1E3A8A', cursor: 'pointer' }} onClick={() => navigate('/')}>
                    X-Lite Medical
                </Typography>
                
                {user && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Button 
                            color={location.pathname === '/' ? 'primary' : 'inherit'}
                            onClick={() => navigate('/')}
                        >
                            Upload
                        </Button>
                        <Button 
                            color={location.pathname === '/history' ? 'primary' : 'inherit'}
                            onClick={() => navigate('/history')}
                        >
                            History
                        </Button>
                        <Box sx={{ display: 'flex', alignItems: 'center', ml: 2, bgcolor: '#f1f5f9', py: 0.5, px: 2, borderRadius: 2 }}>
                            <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: '#1E3A8A' }}>Dr</Avatar>
                            <Typography variant="body2" sx={{ mr: 2, fontWeight: 500 }}>
                                {user.username}
                            </Typography>
                        </Box>
                        <Button variant="outlined" color="error" size="small" onClick={handleLogout}>
                            Sign Out
                        </Button>
                    </Box>
                )}
            </Toolbar>
        </AppBar>
    );
}
