import React from 'react';
import { Box, CircularProgress, Typography, Stack } from '@mui/material';

const LoadingSpinner = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        py: 8,
      }}
    >
      <Stack alignItems="center" spacing={2}>
        <CircularProgress
          size={60}
          sx={{
            color: '#667eea',
          }}
        />
        <Typography variant="h6" sx={{ color: '#333' }}>
          Analyzing your chest X-ray...
        </Typography>
        <Typography variant="body2" color="textSecondary">
          This may take a moment
        </Typography>
      </Stack>
    </Box>
  );
};

export default LoadingSpinner;
