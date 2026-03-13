import React, { useState } from 'react';
import { Container, Paper, Box } from '@mui/material';
import './App.css';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import PredictionResults from './components/PredictionResults';
import LoadingSpinner from './components/LoadingSpinner';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setError(null);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Upload image
      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload image');
      }

      const uploadData = await uploadResponse.json();
      const uploadedFilename = uploadData?.data?.filename;
      if (!uploadedFilename) {
        throw new Error('Upload response missing filename');
      }

      setUploadedImage({
        file: file,
        preview: URL.createObjectURL(file),
        filename: uploadedFilename,
      });

      // Get prediction
      const predictResponse = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: uploadedFilename,
          return_heatmap: true,
          confidence_threshold: 0.5,
        }),
      });

      if (!predictResponse.ok) {
        throw new Error('Failed to get predictions');
      }

      const predictionData = await predictResponse.json();
      setPredictions(predictionData);
    } catch (err) {
      setError(err.message || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setUploadedImage(null);
    setPredictions(null);
    setError(null);
  };

  return (
    <div className="app">
      <Header />
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
          {error && (
            <Box
              sx={{
                mb: 2,
                p: 2,
                backgroundColor: '#ffebee',
                color: '#c62828',
                borderRadius: 1,
              }}
            >
              {error}
            </Box>
          )}

          {loading ? (
            <LoadingSpinner />
          ) : predictions ? (
            <PredictionResults
              image={uploadedImage}
              predictions={predictions}
              onReset={handleReset}
            />
          ) : (
            <ImageUpload onUpload={handleImageUpload} />
          )}
        </Paper>
      </Container>
    </div>
  );
}

export default App;
