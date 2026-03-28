import React from 'react';
import { Box, Typography, Button, Card, Stack, Chip } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';

export default function Home() {
    const navigate = useNavigate();

    return (
        <Box sx={{ maxWidth: 1200, mx: 'auto', pt: { xs: 2, md: 4 }, px: 3 }}>
            <Card
                sx={{
                    p: { xs: 3, md: 5 },
                    overflow: 'hidden',
                    position: 'relative',
                    background:
                        'linear-gradient(125deg, rgba(255,255,255,0.98) 0%, rgba(240,249,255,0.98) 55%, rgba(224,242,254,0.95) 100%)',
                }}
            >
                <Box
                    sx={{
                        position: 'absolute',
                        right: -70,
                        top: -60,
                        width: 280,
                        height: 280,
                        borderRadius: '50%',
                        background: 'radial-gradient(circle, rgba(56,189,248,0.24), rgba(56,189,248,0.0) 68%)',
                    }}
                />

                <Stack direction={{ xs: 'column', md: 'row' }} spacing={4} alignItems="center" justifyContent="space-between">
                    <Box sx={{ maxWidth: 620, zIndex: 1 }}>
                        <Chip label="AI-Assisted Radiology Workflow" sx={{ mb: 2, bgcolor: '#e0f2fe', color: '#075985', fontWeight: 700 }} />
                        <Typography variant="h4" sx={{ color: '#0f2f78', mb: 1.5 }}>
                            Faster Chest X-Ray Review with Clinical-Style AI Support
                        </Typography>
                        <Typography variant="body1" sx={{ color: '#334155', mb: 3, lineHeight: 1.8 }}>
                            Upload chest radiographs, review model findings, inspect Grad-CAM overlays, and export structured PDF reports.
                            The interface is designed for clear medical decision support and rapid triage communication.
                        </Typography>

                        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                            <Button
                                variant="contained"
                                startIcon={<CloudUploadIcon />}
                                onClick={() => navigate('/upload')}
                                sx={{ px: 3.2, py: 1.2, bgcolor: '#1d4ed8' }}
                            >
                                Start Analysis
                            </Button>
                            <Button
                                variant="outlined"
                                startIcon={<DescriptionIcon />}
                                onClick={() => navigate('/instructions')}
                                sx={{ px: 3.2, py: 1.2, borderColor: '#1d4ed8', color: '#1d4ed8' }}
                            >
                                View Instructions
                            </Button>
                        </Stack>
                    </Box>

                    <Box
                        sx={{
                            width: { xs: '100%', md: 380 },
                            height: { xs: 300, md: 360 },
                            borderRadius: 4,
                            border: '1px solid #bfdbfe',
                            backgroundColor: '#ffffff',
                            boxShadow: '0 20px 40px rgba(29, 78, 216, 0.12)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            p: 2,
                        }}
                    >
                        <img
                            src="/chest-xray-hero.png"
                            alt="Chest X-ray visual"
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        />
                    </Box>
                </Stack>
            </Card>
        </Box>
    );
}
