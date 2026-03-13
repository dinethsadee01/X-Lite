import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, Card, Grid, Button, Divider, Alert, CircularProgress, Chip, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemText } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ListAltIcon from '@mui/icons-material/ListAlt';
import axios from 'axios';

const RiskChip = ({ risk }) => {
    let color = 'default';
    if (risk === 'High') color = 'error';
    if (risk === 'Medium') color = 'warning';
    if (risk === 'Low') color = 'success';
    
    return <Chip label={`${risk} Risk`} color={color} size="small" />;
};

export default function Results() {
    const location = useLocation();
    const navigate = useNavigate();
    const resultData = location.state?.resultData;
    const [downloading, setDownloading] = useState(false);
    const [openDialog, setOpenDialog] = useState(false);

    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
    const baseUrl = apiUrl.replace('/api', ''); // To fetch static files /static/...

    if (!resultData) {
        return (
            <Box mt={5} textAlign="center">
                <Alert severity="warning">No results found. Please upload an image first.</Alert>
                <Button onClick={() => navigate('/')} sx={{ mt: 2 }}>Go to Dashboard</Button>
            </Box>
        );
    }

    const { originalImage, predictions, heatmap_path, record_id, uploadedFilename } = resultData;
    
    // Sort predictions: highest prob first
    const sortedPredictions = [...predictions].sort((a,b) => b.probability - a.probability);
    const significantFindings = sortedPredictions.filter(p => p.probability >= 0.5); // Or whatever threshold

    const handleDownloadReport = async () => {
        setDownloading(true);
        const token = localStorage.getItem('token');
        try {
            // First, trigger PDF generation in backend
            const payload = {
                patient_id: "Internal-System-User",
                predictions: predictions,
                image_filename: uploadedFilename,
                record_id: record_id || null,
                additional_notes: "Auto-generated report from web portal."
            };
            
            const pdfGenRes = await axios.post(`${apiUrl}/report/generate`, payload, {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            // Then download it
            const downloadUrl = `${apiUrl.replace('/api', '')}${pdfGenRes.data.download_url}`;
            
            // Fetch as blob to trigger download prompt correctly mapped with JWT although download endpoints usually don't need it if we append token to URL. Standard approach is fetching blob.
            const response = await axios.get(downloadUrl, {
                responseType: 'blob',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', pdfGenRes.data.report_path.split('/').pop());
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);

        } catch (error) {
            console.error("Failed to generate/download report", error);
            alert("Failed to download PDF report. Try again.");
        } finally {
            setDownloading(false);
        }
    };

    return (
        <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/')} sx={{ mb: 1 }}>
                        Back to Upload
                    </Button>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1E3A8A' }}>Diagnostic Results</Typography>
                </Box>
                <Button 
                    variant="contained" 
                    color="primary" 
                    startIcon={downloading ? <CircularProgress size={20} color="inherit" /> : <DownloadIcon />}
                    onClick={handleDownloadReport}
                    disabled={downloading}
                    sx={{ bgcolor: '#1E3A8A', '&:hover': { bgcolor: '#1e40af' } }}
                >
                    {downloading ? 'Preparing PDF...' : 'Download Clinical PDF'}
                </Button>
            </Box>

            <Grid container spacing={4}>
                {/* Image Section */}
                <Grid item xs={12} md={7}>
                    <Card sx={{ p: 2, bgcolor: '#f8fafc', boxShadow: 2 }}>
                        <Typography variant="h6" mb={2} color="#475569">Imaging Visualization</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={6}>
                                <Typography variant="subtitle2" align="center" mb={1}>Original Radiograph</Typography>
                                <img src={originalImage} alt="Original X-Ray" style={{ width: '100%', borderRadius: '8px', border: '1px solid #cbd5e1' }} />
                            </Grid>
                            <Grid item xs={6}>
                                <Typography variant="subtitle2" align="center" mb={1}>Grad-CAM Overlay</Typography>
                                {heatmap_path ? (
                                    <img src={`${baseUrl}${heatmap_path}`} alt="Heatmap" style={{ width: '100%', borderRadius: '8px', border: '1px solid #cbd5e1' }} />
                                ) : (
                                    <Box sx={{ width: '100%', paddingTop: '100%', bgcolor: '#e2e8f0', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                                        <Typography sx={{ position: 'absolute', top: '45%' }} color="text.secondary">Heatmap not available</Typography>
                                    </Box>
                                )}
                            </Grid>
                        </Grid>
                    </Card>
                </Grid>

                {/* Findings Section */}
                <Grid item xs={12} md={5}>
                    <Card sx={{ p: 3, height: '100%', boxShadow: 2 }}>
                        <Typography variant="h6" mb={2} color="#1E3A8A" fontWeight="bold">Key Findings</Typography>
                        <Divider sx={{ mb: 2 }} />
                        
                        {significantFindings.length > 0 ? (
                            <List disablePadding>
                                {significantFindings.map((finding, idx) => (
                                    <ListItem key={idx} sx={{ px: 0, py: 1.5, borderBottom: '1px solid #f1f5f9' }}>
                                        <ListItemText 
                                            primary={<Typography variant="subtitle1" fontWeight="bold">{finding.disease.replace('_', ' ')}</Typography>}
                                            secondary={`Confidence: ${(finding.probability * 100).toFixed(1)}%`}
                                        />
                                        <RiskChip risk={finding.risk_level.charAt(0).toUpperCase() + finding.risk_level.slice(1)} />
                                    </ListItem>
                                ))}
                            </List>
                        ) : (
                            <Alert severity="success" sx={{ mb: 3 }}>No significant findings detected. (All confidence levels &lt; 50%)</Alert>
                        )}
                        
                        <Box mt={3} pt={2} borderTop="1px solid #e2e8f0">
                            <Button 
                                variant="outlined" 
                                fullWidth 
                                startIcon={<ListAltIcon />}
                                onClick={() => setOpenDialog(true)}
                            >
                                View Complete Feature Rankings
                            </Button>
                        </Box>
                    </Card>
                </Grid>
            </Grid>

            {/* Dialog for all rankings */}
            <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ fontWeight: 'bold', color: '#1E3A8A' }}>Complete Feature Analysis (15 Classes)</DialogTitle>
                <DialogContent dividers>
                    <List disablePadding>
                        {sortedPredictions.map((pred, i) => (
                            <ListItem key={i} sx={{ borderBottom: '1px solid #f1f5f9' }}>
                                <ListItemText 
                                    primary={pred.disease.replace('_', ' ')}
                                    secondary={`${(pred.probability * 100).toFixed(2)}% chance`}
                                />
                                <RiskChip risk={pred.risk_level.charAt(0).toUpperCase() + pred.risk_level.slice(1)} />
                            </ListItem>
                        ))}
                    </List>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenDialog(false)}>Close</Button>
                </DialogActions>
            </Dialog>

        </Box>
    );
}
