import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Button, Stack } from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import './ImageUpload.css';

const ImageUpload = ({ onUpload }) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0]);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    multiple: false,
  });

  return (
    <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
      <input {...getInputProps()} />
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          py: 6,
        }}
      >
        <CloudUpload
          sx={{
            fontSize: 64,
            color: '#667eea',
            mb: 2,
          }}
        />
        <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>
          {isDragActive
            ? 'Drop your chest X-ray image here'
            : 'Drag & drop your chest X-ray image here'}
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          or click to select a file (JPEG, PNG)
        </Typography>
        <Button
          variant="contained"
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          }}
        >
          Select Image
        </Button>
      </Box>
    </div>
  );
};

export default ImageUpload;
