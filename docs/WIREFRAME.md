# X-Lite Application Wireframe (Low Fidelity)

## Overview
This wireframe illustrates the user flow and interface layout for the X-Lite chest X-ray analysis application.

---

## 1. Main Page - Upload State

```
┌─────────────────────────────────────────────────────────────┐
│                       X-LITE                                 │
│           Chest X-Ray Disease Detection                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                    ┌───────────────────┐                     │
│                    │                   │                     │
│                    │   📁  Upload      │                     │
│                    │   Chest X-Ray     │                     │
│                    │                   │                     │
│                    │   Drag & Drop     │                     │
│                    │       or          │                     │
│                    │  [Choose File]    │                     │
│                    │                   │                     │
│                    └───────────────────┘                     │
│                                                              │
│         Supported: PNG, JPG, JPEG, DCM                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Loading State

```
┌─────────────────────────────────────────────────────────────┐
│                       X-LITE                                 │
│           Chest X-Ray Disease Detection                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                                                              │
│                        ⌛ Loading...                         │
│                                                              │
│                  Analyzing X-Ray Image                       │
│                                                              │
│                     ▓▓▓▓▓▓▓▓▓░░░                            │
│                                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Results Display State

```
┌─────────────────────────────────────────────────────────────┐
│                       X-LITE                                 │
│           Chest X-Ray Disease Detection                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  [Reset] [Download Report]                                   │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│  ┌────────────────────────┐  │  PREDICTION RESULTS          │
│  │                        │  │                              │
│  │                        │  │  ✓ Positive Findings:        │
│  │   X-RAY IMAGE          │  │  • Pneumonia (High Risk)     │
│  │   (Original)           │  │  • Infiltration (Med Risk)   │
│  │                        │  │                              │
│  │                        │  │  Disease Probabilities:      │
│  └────────────────────────┘  │                              │
│                              │  ┌─────────────────────────┐ │
│  ┌────────────────────────┐  │  │ Pneumonia      89.3% ▓▓ │ │
│  │                        │  │  └─────────────────────────┘ │
│  │   GRAD-CAM             │  │  ┌─────────────────────────┐ │
│  │   HEATMAP              │  │  │ Infiltration   67.2% ▓▓ │ │
│  │                        │  │  └─────────────────────────┘ │
│  │  (Highlighted areas)   │  │  ┌─────────────────────────┐ │
│  │                        │  │  │ Effusion       45.1% ▓░ │ │
│  └────────────────────────┘  │  └─────────────────────────┘ │
│                              │  ┌─────────────────────────┐ │
│                              │  │ Atelectasis    32.7% ▓░ │ │
│                              │  └─────────────────────────┘ │
│                              │  ┌─────────────────────────┐ │
│                              │  │ Cardiomegaly   28.4% ░░ │ │
│                              │  └─────────────────────────┘ │
│                              │                              │
│                              │  Legend:                     │
│                              │  🔴 High Risk (>70%)         │
│                              │  🟡 Medium Risk (50-70%)     │
│                              │  🟢 Low Risk (<50%)          │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 4. Error State

```
┌─────────────────────────────────────────────────────────────┐
│                       X-LITE                                 │
│           Chest X-Ray Disease Detection                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   ╔════════════════════════════════════════════════════╗    │
│   ║  ⚠ Error                                           ║    │
│   ║                                                    ║    │
│   ║  Failed to upload image. Please try again.        ║    │
│   ║  Supported formats: PNG, JPG, JPEG, DCM           ║    │
│   ║                                                    ║    │
│   ╚════════════════════════════════════════════════════╝    │
│                                                              │
│                    [Try Again]                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. PDF Report Layout

```
┌─────────────────────────────────────────────────────────────┐
│  X-LITE DIAGNOSTIC REPORT                                    │
│  Generated: [Date/Time]                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PATIENT IMAGE                                               │
│  ┌──────────────────┐                                        │
│  │                  │                                        │
│  │   X-Ray Image    │                                        │
│  │                  │                                        │
│  └──────────────────┘                                        │
│                                                              │
│  POSITIVE FINDINGS:                                          │
│  • Pneumonia - 89.3% (High Risk)                             │
│  • Infiltration - 67.2% (Medium Risk)                        │
│                                                              │
│  COMPLETE ANALYSIS:                                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Disease          | Probability | Risk Level            ││
│  ├─────────────────────────────────────────────────────────┤│
│  │ Pneumonia        | 89.3%       | High                  ││
│  │ Infiltration     | 67.2%       | Medium                ││
│  │ Effusion         | 45.1%       | Low                   ││
│  │ Atelectasis      | 32.7%       | Low                   ││
│  │ Cardiomegaly     | 28.4%       | Low                   ││
│  │ ...              | ...         | ...                   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  GRAD-CAM VISUALIZATION:                                     │
│  ┌──────────────────┐                                        │
│  │                  │                                        │
│  │   Heatmap        │                                        │
│  │                  │                                        │
│  └──────────────────┘                                        │
│                                                              │
│  DISCLAIMER:                                                 │
│  This report is generated by AI and should be reviewed       │
│  by a qualified medical professional.                        │
└─────────────────────────────────────────────────────────────┘
```

---

## User Flow Diagram

```
    START
      │
      ▼
┌──────────┐
│ Upload   │
│ X-Ray    │
└─────┬────┘
      │
      ▼
┌──────────┐     Error      ┌──────────┐
│ Loading  │ ─────────────► │  Error   │
│ Analysis │                │  Message │
└─────┬────┘                └─────┬────┘
      │                           │
      │ Success                   │
      ▼                           │
┌──────────┐                      │
│ Display  │                      │
│ Results  │                      │
└─────┬────┘                      │
      │                           │
      ├──► [Download Report]      │
      │                           │
      └──► [Reset] ───────────────┘
            │
            ▼
         Upload
```

---

## Component Breakdown

### Header Component
- Application title
- Subtitle/tagline
- Navigation (if needed)

### ImageUpload Component
- Drag-and-drop zone
- File picker button
- File type validation
- Preview thumbnail (optional)

### LoadingSpinner Component
- Animated loading indicator
- Progress message
- Optional progress bar

### PredictionResults Component
- Two-column layout:
  - **Left**: Original image + Grad-CAM heatmap
  - **Right**: Prediction results list
- Positive findings highlight
- Disease probability bars
- Risk level indicators (color-coded)
- Action buttons (Reset, Download Report)

### Error Display
- Error message container
- Retry button
- User-friendly error descriptions

---

## Color Scheme

```
Risk Levels:
🔴 High Risk:   #c62828 (Red)
🟡 Medium Risk: #f9a825 (Yellow/Amber)
🟢 Low Risk:    #2e7d32 (Green)

UI Elements:
Primary:   #1976d2 (Blue)
Secondary: #424242 (Gray)
Background: #f5f5f5 (Light Gray)
Paper:     #ffffff (White)
Error:     #d32f2f (Red)
```

---

## Responsive Behavior

### Desktop (>960px)
- Two-column layout for results
- Full-size images and heatmaps
- Side-by-side comparison

### Tablet (600-960px)
- Stacked layout
- Reduced image sizes
- Maintained readability

### Mobile (<600px)
- Single column layout
- Collapsible sections
- Touch-optimized buttons
- Scrollable results

---

## API Integration Points

1. **POST /api/upload**
   - Upload X-ray image
   - Returns: filename, upload status

2. **POST /api/predict**
   - Request prediction
   - Parameters: filename, return_heatmap, confidence_threshold
   - Returns: predictions, heatmap data

3. **GET /api/report**
   - Generate PDF report
   - Parameters: filename, predictions
   - Returns: PDF file

4. **GET /api/health**
   - Health check endpoint

---

## Notes

- All disease labels (14 total): Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- Confidence threshold: 0.5 (50%) by default
- Heatmap overlay uses transparency for better visualization
- Results sorted by probability (highest first)
- PDF report includes timestamp and disclaimer

