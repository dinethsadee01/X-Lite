import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  LinearProgress,
  Chip,
} from '@mui/material';
import { CheckCircle, Warning } from '@mui/icons-material';

const PredictionResults = ({ image, predictions, onReset }) => {
  const positiveFindings = predictions.positive_findings || [];
  const allPredictions = predictions.predictions || [];

  // Sort predictions by probability
  const sortedPredictions = [...allPredictions].sort(
    (a, b) => b.probability - a.probability
  );

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high':
        return '#d32f2f';
      case 'medium':
        return '#fbc02d';
      case 'low':
        return '#388e3c';
      default:
        return '#1976d2';
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high':
        return '#ffebee';
      case 'medium':
        return '#fffde7';
      case 'low':
        return '#e8f5e9';
      default:
        return '#e3f2fd';
    }
  };

  return (
    <Box>
      {/* Image Display */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ p: 0 }}>
              <Box
                component="img"
                src={image.preview}
                alt="X-ray"
                sx={{
                  width: '100%',
                  height: 'auto',
                  borderRadius: '4px 4px 0 0',
                  display: 'block',
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Summary */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Summary
              </Typography>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Processing Time
                  </Typography>
                  <Typography variant="h6">
                    {predictions.processing_time_ms?.toFixed(2)} ms
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Model Used
                  </Typography>
                  <Typography variant="h6">{predictions.model_name}</Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Findings Detected
                  </Typography>
                  <Typography
                    variant="h6"
                    sx={{
                      color: positiveFindings.length > 0 ? '#d32f2f' : '#388e3c',
                    }}
                  >
                    {predictions.num_positive || 0} condition(s)
                  </Typography>
                </Box>
                {positiveFindings.length > 0 && (
                  <Box>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                      Positive Findings:
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {positiveFindings.map((finding, idx) => (
                        <Chip
                          key={idx}
                          label={finding}
                          size="small"
                          icon={<Warning sx={{ fontSize: '18px !important' }} />}
                          sx={{
                            backgroundColor: '#ffebee',
                            color: '#d32f2f',
                            fontWeight: 'bold',
                          }}
                        />
                      ))}
                    </Stack>
                  </Box>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Predictions */}
      <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
        Detailed Predictions
      </Typography>
      <Stack spacing={2} sx={{ mb: 4 }}>
        {sortedPredictions.map((pred, idx) => (
          <Card key={idx}>
            <CardContent>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 2,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  {pred.probability > 0.5 && (
                    <CheckCircle sx={{ color: getRiskColor(pred.risk_level) }} />
                  )}
                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      {pred.disease}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {pred.description}
                    </Typography>
                  </Box>
                </Box>
                <Chip
                  label={pred.risk_level}
                  size="small"
                  sx={{
                    backgroundColor: getRiskBgColor(pred.risk_level),
                    color: getRiskColor(pred.risk_level),
                    fontWeight: 'bold',
                  }}
                />
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{ flex: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={pred.probability * 100}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getRiskColor(pred.risk_level),
                      },
                    }}
                  />
                </Box>
                <Typography
                  variant="body2"
                  sx={{
                    minWidth: 50,
                    textAlign: 'right',
                    fontWeight: 'bold',
                  }}
                >
                  {(pred.probability * 100).toFixed(1)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Stack>

      {/* Action Buttons */}
      <Stack direction="row" spacing={2} sx={{ justifyContent: 'flex-end' }}>
        <Button
          variant="outlined"
          onClick={onReset}
          sx={{
            borderColor: '#667eea',
            color: '#667eea',
            '&:hover': {
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
            },
          }}
        >
          Analyze Another Image
        </Button>
      </Stack>
    </Box>
  );
};

export default PredictionResults;
