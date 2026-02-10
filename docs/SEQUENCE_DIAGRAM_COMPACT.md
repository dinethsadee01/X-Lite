# X-Lite Compact Sequence Diagram

## Overview
This is a simplified version of the X-Lite system workflow, optimized for presentation slides. It shows the three main user workflows in a condensed format.

---

## Simplified Components

### Participants (5 total - reduced from 8)
1. **User**: End user/clinician
2. **Frontend (UI)**: React web application
3. **Backend API**: FastAPI server (combines API + Services)
4. **ML Model**: Hybrid CNN-Transformer (combines Prediction Service + Model)
5. **Storage**: File system (combines uploads + checkpoints + cache)

---

## Three Main Workflows

### 1️⃣ Upload X-Ray (6 steps)
```
User → Frontend → Backend API → Storage → Backend API → Frontend
```

**Flow**:
1. User selects chest X-ray image
2. Frontend sends file to Backend API
3. Backend API saves image to Storage
4. Storage returns filename
5. Backend API confirms upload success
6. Frontend updates UI

**Time**: ~500ms - 2s (depends on file size and network)

---

### 2️⃣ Get Prediction (11 steps)
```
Frontend → Backend API → Storage → Backend API → ML Model → Backend API → Frontend → User
```

**Flow**:
1. Frontend requests prediction
2. Backend API loads image from Storage
3. Backend API preprocesses image (CLAHE, resize, normalize)
4. Storage returns preprocessed image data
5. Backend API sends to ML Model
6. ML Model runs inference (CNN + Attention + Classification)
7. ML Model returns disease probabilities
8. Backend API requests Grad-CAM from ML Model
9. ML Model generates heatmap
10. Backend API sends results + heatmap to Frontend
11. Frontend displays results to User

**Time**: <1 second

**Output**:
- 14 disease probabilities
- Risk levels (High/Medium/Low)
- Grad-CAM heatmap
- Positive findings list

---

### 3️⃣ Generate Report (8 steps)
```
User → Frontend → Backend API → Storage → Backend API → Frontend → User
```

**Flow**:
1. User clicks "Download Report"
2. Frontend requests PDF generation
3. Backend API loads original image from Storage
4. Backend API creates PDF document
5. Backend API saves PDF to Storage
6. Backend API sends PDF file to Frontend
7. Frontend triggers browser download
8. User receives diagnostic report PDF

**Time**: ~500ms

**PDF Contents**:
- Timestamp and header
- Original X-ray image
- Positive findings
- Complete analysis table
- Grad-CAM visualization
- Medical disclaimer

---

## Key Simplifications

### Combined Components
- **Backend API**: Merged API Gateway + Service Layer (Image Service, Prediction Service, Report Service)
- **ML Model**: Combined Prediction Service + Student Model + Grad-CAM
- **Storage**: Unified File Storage + Model Checkpoints + CLAHE Cache

### Removed Details
- ❌ Detailed preprocessing steps
- ❌ Individual service method calls
- ❌ Error handling flows
- ❌ Validation steps
- ❌ Configuration dependencies
- ❌ Teacher model (only used in training)
- ❌ Loading/initialization on first request

### Consolidated Steps
- Image preprocessing shown as single step
- ML inference shown as single operation with internal detail
- PDF creation shown as atomic operation

---

## Presentation Tips

### Slide Layout Suggestion

**Title Slide**:
```
"X-Lite System Workflow"
Subtitle: Automated Chest X-Ray Analysis
```

**Main Diagram Slide**:
- **Top**: Brief text intro (2 lines)
- **Center**: Compact sequence diagram
- **Bottom**: Key metrics box
  ```
  ⏱️ Upload: 500ms - 2s  |  🧠 Inference: <1s  |  📄 Report: 500ms
  ```

**Appendix Slide** (optional):
- Title: "Detailed Sequence Diagram"
- Full detailed diagram from `sequence_diagram.mmd`
- Reference slide for technical audience

---

## One-Liner Descriptions

Use these for slide annotations:

1. **Upload**: "User uploads chest X-ray image to system"
2. **Prediction**: "AI analyzes image and identifies 14 possible diseases"
3. **Report**: "System generates downloadable PDF diagnostic report"

---

## Statistics to Highlight

### Model Performance
- **Accuracy**: Competitive with CheXNet baseline
- **Speed**: <1 second inference on CPU
- **Size**: 5-25M parameters (lightweight)

### System Capabilities
- **Multi-Label**: Detects 14 thoracic diseases simultaneously
- **Explainability**: Grad-CAM heatmaps show model reasoning
- **Resource-Efficient**: Runs on standard computers (no GPU needed)

---

## Color Coding for PPT

### Suggested Color Palette
- **User/Frontend**: Blue (#1976d2)
- **Backend API**: Orange (#f57c00)
- **ML Model**: Red/Pink (#c2185b)
- **Storage**: Yellow (#f9a825)

### Arrow Colors
- **Request**: Solid dark arrow
- **Response**: Dashed return arrow
- **Internal**: Dotted self-arrow

---

## Animation Sequence (PowerPoint)

If animating the diagram:

1. **Appear**: Show all participants first
2. **Wipe**: Section 1 note ("1. Upload X-Ray")
3. **Fly In**: Arrow sequence for upload flow
4. **Wipe**: Section 2 note ("2. Get Prediction")
5. **Fly In**: Arrow sequence for prediction flow
6. **Wipe**: Section 3 note ("3. Generate Report")
7. **Fly In**: Arrow sequence for report flow

**Timing**: 0.3s per arrow, 1s pause at each section

---

## Comparison: Compact vs. Full Diagram

| Aspect | Compact | Full |
|--------|---------|------|
| **Participants** | 5 | 8 |
| **Workflows** | 3 | 4 (includes Reset) |
| **Total Steps** | ~25 | ~80 |
| **Complexity** | Low | High |
| **Detail Level** | High-level | Implementation |
| **Use Case** | Presentations | Documentation |
| **Audience** | General/Business | Technical |

---

## Usage Instructions

### For Mermaid Live Editor
1. Go to https://mermaid.live
2. Paste contents of `sequence_diagram_compact.mmd`
3. Export as PNG/SVG for PowerPoint
4. Recommended: SVG for scalability

### For VS Code
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file or the `.mmd` file
3. Use preview to view diagram
4. Screenshot or export

### For PowerPoint
1. Export diagram as SVG or high-res PNG
2. Insert as picture
3. Add text boxes for annotations
4. Apply slide master formatting

---

## Key Talking Points

When presenting this diagram:

### Slide 1: Introduction
- "X-Lite streamlines chest X-ray analysis with AI"
- "Three simple steps from upload to diagnosis"

### Slide 2: Technical Overview
- "Lightweight hybrid CNN-Transformer architecture"
- "Knowledge distillation from expert teacher model"
- "CPU-optimized for resource-limited settings"

### Slide 3: Clinical Value
- "Assists radiologists with automated screening"
- "Visual explanations build trust with Grad-CAM"
- "PDF reports integrate with clinical workflow"

### Slide 4: Future Roadmap
- "Multi-modal inputs (patient metadata)"
- "PACS integration for hospital deployment"
- "Mobile application for point-of-care use"

---

## Related Files

- **Compact Diagram**: [sequence_diagram_compact.mmd](sequence_diagram_compact.mmd)
- **Full Diagram**: [sequence_diagram.mmd](sequence_diagram.mmd)
- **Documentation**: [SEQUENCE_DIAGRAM.md](SEQUENCE_DIAGRAM.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Export Formats

### Recommended Exports

**For PowerPoint**:
- Format: SVG (vector, scalable)
- Resolution: N/A (vector)
- Background: Transparent

**For PDF/Print**:
- Format: PNG
- Resolution: 300 DPI
- Background: White

**For Web/Documentation**:
- Format: SVG or WebP
- Optimization: Compressed
- Background: Transparent

---

## Accessibility Notes

Ensure slide meets accessibility standards:
- **Contrast**: Dark text on light background
- **Font Size**: Minimum 18pt for labels
- **Alt Text**: Describe workflow in image properties
- **Color**: Don't rely solely on color (use patterns/labels)

---

## License & Attribution

When using in presentations:
- **Credit**: "X-Lite: Lightweight Chest X-Ray Analysis System"
- **Authors**: [Your name/institution]
- **Dataset**: ChestX-ray14 (NIH Clinical Center)
- **Date**: February 2026
