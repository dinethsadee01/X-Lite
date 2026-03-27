import React from 'react';
import { Box, Typography, Card, Stack, Chip } from '@mui/material';

const steps = [
    {
        title: '1. Upload Chest X-Ray',
        text: 'Go to Upload and drag-drop or browse for a chest X-ray image (JPG/PNG).',
    },
    {
        title: '2. Run Diagnostics',
        text: 'Click Run Diagnostics. The model preprocesses the image and performs 15-class inference.',
    },
    {
        title: '3. Review Results',
        text: 'Open the results screen to view disease confidence ranking, risk tags, and heatmap output when available.',
    },
    {
        title: '4. Download Clinical PDF',
        text: 'Use the Download Clinical PDF button to export a structured report for sharing or archive.',
    },
    {
        title: '5. Re-check History',
        text: 'From History, review previously analyzed sessions (or placeholder data in demo mode).',
    },
];

export default function Instructions() {
    return (
        <Box sx={{ maxWidth: 1000, mx: 'auto', pt: { xs: 14, md: 17 }, px: 3 }}>
            <Typography variant="h4" sx={{ color: '#1e3a8a', mb: 1.2 }}>
                Usage Instructions
            </Typography>
            <Typography variant="body1" sx={{ color: '#475569', mb: 3 }}>
                Follow this workflow for consistent and clinically readable outputs.
            </Typography>

            <Stack spacing={2.2}>
                {steps.map((step) => (
                    <Card key={step.title} sx={{ p: 2.4, backgroundColor: '#ffffff' }}>
                        <Chip label={step.title} sx={{ mb: 1.2, bgcolor: '#eff6ff', color: '#1e40af', fontWeight: 700 }} />
                        <Typography variant="body1" sx={{ color: '#334155', lineHeight: 1.8 }}>
                            {step.text}
                        </Typography>
                    </Card>
                ))}
            </Stack>
        </Box>
    );
}
