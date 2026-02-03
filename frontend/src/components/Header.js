import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import { Favorite } from '@mui/icons-material';

const Header = () => {
  return (
    <AppBar position="static" sx={{ background: 'rgba(0, 0, 0, 0.9)' }}>
      <Toolbar>
        <Favorite sx={{ mr: 1, color: '#ff6b6b' }} />
        <Typography
          variant="h5"
          component="h1"
          sx={{
            flexGrow: 1,
            fontWeight: 'bold',
            letterSpacing: 1,
          }}
        >
          X-Lite
        </Typography>
        <Typography variant="caption" sx={{ opacity: 0.8 }}>
          Chest X-Ray Classification
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
