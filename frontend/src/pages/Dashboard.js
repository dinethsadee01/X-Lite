import React, { useState, useCallback } from 'react';
import { Box, Typography, Button, Paper, Alert, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from 'axios';

export default function Dashboard() {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState('');
    const [resultData, setResultData] = useState(null);

    const navigate = useNavigate();
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0]);
            setError('');
            setResultData(null);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/png': ['.png']
        },
        maxFiles: 1
    });

    const handleUploadAndPredict = async () => {
        if (!file) return;
        
        setError('');
        setUploading(true);
        const token = localStorage.getItem('token');
        
        try {
            // 1. Upload File
            const formData = new FormData();
            formData.append('file', file);
            const uploadRes = await axios.post(`${apiUrl}/upload`, formData, {
                headers: { 
                    'Content-Type': 'multipart/form-data',
                    'Authorization': `Bearer ${token}`
                }
            });

            const uploadedFilename = uploadRes.data.data.filename;
            setUploading(false);
            setAnalyzing(true);

            // 2. Predict
            const predictRes = await axios.post(`${apiUrl}/predict`, {
                filename: uploadedFilename,
                return_heatmap: true
            }, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            console.log(predictRes.data);
            setResultData({
                ...predictRes.data,
                originalImage: URL.createObjectURL(file), // Local preview
                uploadedFilename
            });
            setAnalyzing(false);

        } catch (err) {
            setUploading(false);
            setAnalyzing(false);
            setError(err.response?.data?.detail || 'An error occurred during analysis');
        }
    };

    return (
        <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, p: 3 }}>
            <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1E3A8A', mb: 1 }}>
                Clinical X-Ray Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary" mb={4}>
                Upload a chest radiograph to run our 15-class inference engine.
            </Typography>

            <Paper 
                {...getRootProps()} 
                sx={{
                    p: 6,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: isDragActive ? '#e0f2fe' : '#f8fafc',
                    border: '2px dashed',
                    borderColor: isDragActive ? '#3b82f6' : '#cbd5e1',
                    borderRadius: 3,
                    transition: 'all 0.2s ease',
                    '&:hover': { bgcolor: '#f1f5f9', borderColor: '#94a3b8' }
                }}
            >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 64, color: '#94a3b8', mb: 2 }} />
                
                {file ? (
                    <Box>
                        <Typography variant="h6" color="success.main" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                            <CheckCircleIcon /> File selected: {file.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" mt={1}>
                            Click or drag to change file
                        </Typography>
                    </Box>
                ) : (
                    <Box>
                        <Typography variant="h6" color="text.primary">
                            Drag & drop patient X-ray here
                        </Typography>
                        <Typography variant="body2" color="text.secondary" mt={1}>
                            or click to browse local files (JPG, PNG)
                        </Typography>
                    </Box>
                )}
            </Paper>

            {error && <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>}

            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, gap: 2 }}>
                <Button
                    variant="contained"
                    size="large"
                    onClick={handleUploadAndPredict}
                    disabled={!file || uploading || analyzing}
                    sx={{ bgcolor: '#1E3A8A', '&:hover': { bgcolor: '#1e40af' }, px: 4 }}
                >
                    {uploading ? <CircularProgress size={24} color="inherit" /> : 
                     analyzing ? 'Analyzing X-Ray...' : 'Run Diagnostics'}
                </Button>

                <Button
                    variant="outlined"
                    size="large"
                    disabled={!resultData}
                    onClick={() => navigate('/results', { state: { resultData } })}
                    sx={{ px: 4, borderColor: '#1E3A8A', color: '#1E3A8A' }}
                >
                    View Diagnostic Report
                </Button>
            </Box>
        </Box>
    );
}
