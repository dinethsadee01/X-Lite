import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Chip, Button, IconButton, CircularProgress } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import axios from 'axios';

export default function History() {
    const [records, setRecords] = useState([]);
    const [loading, setLoading] = useState(true);

    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
    const baseUrl = apiUrl.replace('/api', '');

    const fetchHistory = useCallback(async () => {
        try {
            const token = localStorage.getItem('token');
            const res = await axios.get(`${apiUrl}/history`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            setRecords(res.data.data || []);
        } catch (error) {
            console.error("Failed to fetch history:", error);
        } finally {
            setLoading(false);
        }
    }, [apiUrl]);

    useEffect(() => {
        fetchHistory();
    }, [fetchHistory]);

    const handleDelete = async (id) => {
        if (!window.confirm("Are you sure you want to delete this record?")) return;
        try {
            const token = localStorage.getItem('token');
            await axios.delete(`${apiUrl}/history/${id}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            setRecords(records.filter(r => r._id !== id));
        } catch (error) {
            console.error("Failed to delete record:", error);
            alert("Error deleting record.");
        }
    };

    const handleDownloadPdf = async (pdfPath) => {
        if (!pdfPath) {
            alert("Report not generated for this record yet.");
            return;
        }
        
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`${baseUrl}${pdfPath.replace('/static/reports/', '/api/report/download/')}`, {
                responseType: 'blob',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', pdfPath.split('/').pop());
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
        } catch (error) {
            console.error("Failed to download PDF:", error);
            alert("Error downloading PDF.");
        }
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
            <Typography variant="h4" sx={{ color: '#1e3a8a', mb: 3 }}>
                Patient Analysis History
            </Typography>

            <TableContainer component={Paper} elevation={2}>
                <Table sx={{ minWidth: 650 }}>
                    <TableHead sx={{ bgcolor: '#eff6ff' }}>
                        <TableRow>
                            <TableCell sx={{ fontWeight: 'bold' }}>Date & Time</TableCell>
                            <TableCell sx={{ fontWeight: 'bold' }}>Image File</TableCell>
                            <TableCell sx={{ fontWeight: 'bold' }}>Top Finding</TableCell>
                            <TableCell sx={{ fontWeight: 'bold' }}>Risk Severity</TableCell>
                            <TableCell align="center" sx={{ fontWeight: 'bold' }}>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {records.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={5} align="center" sx={{ py: 3 }}>
                                    <Typography color="text.secondary">No analysis history found.</Typography>
                                </TableCell>
                            </TableRow>
                        ) : null}

                        {records.map((record) => {
                            // Find highest probability prediction
                            let topPred = null;
                            if (record.predictions && record.predictions.length > 0) {
                                topPred = [...record.predictions].sort((a,b) => b.probability - a.probability)[0];
                            }

                            const riskColor = topPred?.risk_level?.toLowerCase() === 'high' ? 'error' : 
                                              (topPred?.risk_level?.toLowerCase() === 'medium' ? 'warning' : 'success');

                            return (
                                <TableRow key={record._id} hover>
                                    <TableCell>
                                        {new Date(record.created_at).toLocaleString()}
                                    </TableCell>
                                    <TableCell>{record.filename}</TableCell>
                                    <TableCell>
                                        {topPred ? topPred.disease.replace('_', ' ') : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                        {topPred ? (
                                            <Chip label={topPred.risk_level.toUpperCase()} size="small" color={riskColor} />
                                        ) : 'N/A'}
                                    </TableCell>
                                    <TableCell align="center">
                                        <Button 
                                            size="small" 
                                            variant="outlined"
                                            startIcon={<PictureAsPdfIcon fontSize="small" />}
                                            onClick={() => handleDownloadPdf(record.pdf_report_path)}
                                            disabled={!record.pdf_report_path}
                                            sx={{ mr: 1 }}
                                        >
                                            PDF
                                        </Button>
                                        <IconButton 
                                            color="error" 
                                            size="small"
                                            onClick={() => handleDelete(record._id)}
                                        >
                                            <DeleteIcon fontSize="small" />
                                        </IconButton>
                                    </TableCell>
                                </TableRow>
                            );
                        })}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
}
