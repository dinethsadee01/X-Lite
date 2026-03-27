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

    const navItems = [
        { label: 'Home', path: '/' },
        { label: 'Upload', path: '/upload' },
        { label: 'Instructions', path: '/instructions' },
        { label: 'History', path: '/history' },
    ];

    return (
        <AppBar
            position="fixed"
            color="transparent"
            elevation={0}
            sx={{
                backgroundColor: 'transparent',
                backdropFilter: 'none',
                borderBottom: 'none',
                px: 2,
                pt: 1,
            }}
        >
            <Toolbar sx={{ minHeight: 74, display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <MonitorHeartIcon sx={{ color: '#1d4ed8', mr: 1, fontSize: 28 }} />
                    <Typography variant="h6" component="div" sx={{ fontWeight: 800, color: '#1e3a8a', cursor: 'pointer' }} onClick={() => navigate('/')}>
                    X-Lite Medical
                    </Typography>
                </Box>

                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        px: 1.5,
                        py: 1,
                        borderRadius: 999,
                        border: '1px solid #bfdbfe',
                        backgroundColor: 'rgba(255, 255, 255, 0.82)',
                        backdropFilter: 'blur(10px)',
                        boxShadow: '0 10px 30px rgba(14, 116, 217, 0.10)',
                    }}
                >
                    {navItems.map((item) => {
                        const isActive = location.pathname === item.path;
                        return (
                            <Button
                                key={item.path}
                                onClick={() => navigate(item.path)}
                                variant={isActive ? 'contained' : 'text'}
                                color={isActive ? 'primary' : 'inherit'}
                                sx={{ px: 2.2, borderRadius: 999 }}
                            >
                                {item.label}
                            </Button>
                        );
                    })}
                </Box>
                
                {user && (
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 1.5 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', bgcolor: 'rgba(255,255,255,0.86)', py: 0.5, px: 1.5, borderRadius: 999, border: '1px solid #bfdbfe' }}>
                            <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: '#1d4ed8' }}>Dr</Avatar>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {user.username}
                            </Typography>
                        </Box>
                        <Button variant="outlined" color="error" size="small" onClick={handleLogout} sx={{ borderRadius: 3 }}>
                            Sign Out
                        </Button>
                    </Box>
                )}
            </Toolbar>
        </AppBar>
    );
}
