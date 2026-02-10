# X-Lite System Class Diagram

## Overview

This class diagram illustrates the object-oriented design of the X-Lite chest X-ray analysis system, showing the main classes, their attributes, methods, and relationships using UML notation.

---

## Class Categories

The system is organized into 6 main categories:

1. **API Models** - Request/response data structures
2. **Service Layer** - Business logic services
3. **ML Models** - Neural network architectures
4. **Training Components** - Model training infrastructure
5. **Configuration** - System settings and constants
6. **Explainability** - Grad-CAM visualization

---

## 1. API Models (Pydantic)

### PredictionRequest
**Purpose**: Request model for prediction endpoint

**File**: `backend/routes/predict.py`

**Attributes**:
- `filename: str` - Uploaded image filename
- `return_heatmap: bool` - Whether to generate Grad-CAM (default: True)
- `confidence_threshold: float` - Probability threshold for positive findings (default: 0.5)

**Usage**:
```python
request = PredictionRequest(
    filename="20260210_image.jpg",
    return_heatmap=True,
    confidence_threshold=0.5
)
```

---

### PredictionResult
**Purpose**: Single disease prediction result

**Attributes**:
- `disease: str` - Disease name (e.g., "Pneumonia")
- `probability: float` - Prediction probability (0.0 to 1.0)
- `risk_level: str` - Risk category ("High", "Medium", "Low")
- `color: str` - Hex color for UI display
- `description: str` - Disease description (optional)

**Example**:
```python
result = PredictionResult(
    disease="Pneumonia",
    probability=0.893,
    risk_level="High",
    color="#c62828",
    description="Lung infection causing inflammation..."
)
```

---

### PredictionResponse
**Purpose**: Complete prediction API response

**Attributes**:
- `success: bool` - Request success status
- `predictions: List[PredictionResult]` - All 14 disease predictions
- `positive_findings: List[str]` - Diseases above threshold
- `heatmap_base64: str` - Base64-encoded Grad-CAM image
- `message: str` - Status/error message

**Relationships**:
- **Contains** `PredictionResult` (composition, 1-to-many)

---

## 2. Service Layer

### ImageService
**Purpose**: Image validation, storage, and preprocessing

**File**: `backend/services/image_service.py`

**Attributes**:
- `upload_dir: Path` - Directory for uploaded images

**Methods**:

#### validate_image(file: UploadFile) → bool
Validates uploaded file type and size
- Check file extension against whitelist
- Verify MIME type
- Validate file size limit

#### save_image(file: UploadFile) → str
Saves uploaded image with unique filename
- Generate timestamp + UUID filename
- Save to upload directory
- Return filename

#### load_and_preprocess(filename: str) → Tensor
Loads and preprocesses image for inference
1. Load image from storage
2. Apply CLAHE enhancement
3. Resize to 224×224
4. Normalize pixel values
5. Convert to PyTorch tensor

#### apply_clahe(image: ndarray) → ndarray
Applies Contrast Limited Adaptive Histogram Equalization
- Enhance local contrast
- Preserve image details
- Prevent over-amplification

#### resize_and_normalize(image: ndarray) → Tensor
Resizes image and normalizes pixel values
- Resize to model input size (224×224)
- Normalize to [0, 1] or ImageNet stats
- Convert to tensor format

**Dependencies**: Uses `Config` for settings

---

### PredictionService
**Purpose**: Model inference and prediction logic

**File**: `backend/services/prediction_service.py`

**Attributes**:
- `model: StudentModel` - Loaded ML model (private)
- `device: torch.device` - CPU or GPU device (private)
- `model_loaded: bool` - Model initialization flag (private)

**Methods**:

#### load_model(model_path: str) → void
Loads trained model from checkpoint
- Initialize model architecture
- Load state dict from checkpoint
- Set to evaluation mode
- Move to device (CPU/GPU)

#### predict(image: Tensor, return_heatmap: bool) → dict
Performs model inference
1. Forward pass through model
2. Apply sigmoid activation
3. Compute risk levels
4. Filter by threshold
5. Generate Grad-CAM (if requested)
6. Format response

#### generate_gradcam(image: Tensor, target_class: int) → ndarray
Generates Grad-CAM heatmap for target disease
- Delegate to GradCAM class
- Return heatmap as numpy array

#### assign_risk_level(probability: float) → str
Assigns risk level based on probability
- High: > 0.7
- Medium: 0.5 - 0.7
- Low: < 0.5

#### get_disease_description(disease: str) → str
Retrieves disease description from knowledge base

**Relationships**:
- **Uses** `StudentModel` for inference
- **Uses** `GradCAM` for explainability
- **Creates** `PredictionResponse`
- **Uses** `Config` and `DiseaseLabels`

---

### ReportService
**Purpose**: PDF report generation

**File**: `backend/services/report_service.py`

**Attributes**:
- `reports_dir: Path` - Directory for PDF reports

**Methods**:

#### generate_report(filename: str, predictions: dict) → bytes
Generates complete PDF report
1. Create PDF document
2. Add header and timestamp
3. Add X-ray images
4. Add findings table
5. Add disclaimer
6. Save to storage
7. Return PDF bytes

#### create_pdf_document() → Canvas
Creates new PDF canvas with ReportLab

#### add_header(pdf: Canvas, timestamp: str) → void
Adds report header and metadata

#### add_images(pdf: Canvas, xray: ndarray, heatmap: ndarray) → void
Adds original X-ray and Grad-CAM to PDF

#### add_findings_table(pdf: Canvas, predictions: dict) → void
Adds disease predictions table

#### add_disclaimer(pdf: Canvas) → void
Adds medical disclaimer text

**Dependencies**: Uses `Config` for paths

---

## 3. ML Models

### StudentModel
**Purpose**: Lightweight hybrid CNN-Transformer for inference

**File**: `ml/models/student_model.py`

**Attributes**:
- `backbone: Module` - CNN feature extractor (private)
- `attention: Module` - Attention mechanism (private)
- `classifier: Module` - Classification head (private)
- `num_classes: int` - Number of output classes (14)
- `model_name: str` - Model variant name

**Methods**:

#### forward(x: Tensor) → Tensor
Forward pass through model
1. Extract features via backbone
2. Apply attention mechanism
3. Classify via classification head
4. Return logits (before sigmoid)

#### get_features(x: Tensor) → Tensor
Extract feature maps from backbone
- Used for Grad-CAM
- Returns intermediate representations

#### get_attention_weights() → Tensor
Returns attention weight matrices
- For visualization/analysis
- Shows which regions are important

**Composition**:
- **Has** `ConvNeXtBackbone` (or EfficientNet, MobileNet)
- **Has** `MultiHeadSelfAttention` (or Performer, Linear)
- **Has** `ClassificationHead`

**Inheritance**: Inherits from `torch.nn.Module`

**Variants**:
- `convnext_tiny_mhsa`
- `efficientnet_b0_mhsa`
- `mobilenetv3_linear`

---

### TeacherModel
**Purpose**: High-performance teacher for knowledge distillation

**File**: `ml/models/teacher_model.py`

**Attributes**:
- `backbone: DenseNet121` - Pre-trained DenseNet (private)
- `num_classes: int` - Number of output classes (14)

**Methods**:

#### forward(x: Tensor) → Tensor
Forward pass through DenseNet121
- Returns logits for all 14 diseases

#### load_chexnet_weights(path: str) → void
Loads pre-trained CheXNet weights
- Download from URL if not cached
- Load state dict
- Adapt for 14 classes

**Inheritance**: Inherits from `torch.nn.Module`

**Usage**: Only during training (not loaded in production)

---

### ConvNeXtBackbone
**Purpose**: CNN backbone for feature extraction

**File**: `ml/models/student_model.py` (internal)

**Attributes**:
- `stages: Sequential` - ConvNeXt stages (private)
- `norm: LayerNorm` - Layer normalization (private)
- `out_channels: int` - Output feature dimension

**Methods**:

#### forward(x: Tensor) → Tensor
Extract hierarchical features
- Progressive downsampling
- Depthwise convolutions
- Returns feature maps

**Inheritance**: Inherits from `torch.nn.Module`

**Alternatives**:
- `EfficientNetBackbone`
- `MobileNetV3Backbone`

---

### MultiHeadSelfAttention
**Purpose**: Transformer-style attention mechanism

**File**: Internal to student model

**Attributes**:
- `num_heads: int` - Number of attention heads (private)
- `embed_dim: int` - Embedding dimension (private)
- `qkv_proj: Linear` - Query/key/value projection (private)
- `out_proj: Linear` - Output projection (private)

**Methods**:

#### forward(x: Tensor) → Tensor
Apply multi-head self-attention
1. Project to Q, K, V
2. Compute attention scores
3. Apply softmax
4. Weighted sum of values
5. Project output

#### get_attention_maps() → Tensor
Returns attention weight matrices
- For visualization
- Shows spatial dependencies

**Inheritance**: Inherits from `torch.nn.Module`

**Alternatives**:
- `PerformerAttention` (linear complexity)
- `LinearAttention` (efficient approximation)

---

### ClassificationHead
**Purpose**: Multi-label classification layer

**File**: Internal to student model

**Attributes**:
- `fc1: Linear` - First fully connected layer (private)
- `fc2: Linear` - Second fully connected layer (private)
- `dropout: Dropout` - Dropout regularization (private)
- `num_classes: int` - Number of output classes (14)

**Methods**:

#### forward(x: Tensor) → Tensor
Classify features to disease probabilities
1. Flatten feature maps
2. First linear layer + ReLU
3. Dropout
4. Second linear layer (logits)

**Inheritance**: Inherits from `torch.nn.Module`

---

## 4. Training Components

### KDTrainer
**Purpose**: Knowledge distillation training orchestration

**File**: `ml/training/kd_trainer.py`

**Attributes**:
- `student: StudentModel` - Student model being trained (private)
- `teacher: TeacherModel` - Teacher model (frozen) (private)
- `criterion: DistillationLoss` - Loss function (private)
- `optimizer: Optimizer` - Parameter optimizer (private)
- `train_loader: DataLoader` - Training data (private)
- `val_loader: DataLoader` - Validation data (private)
- `best_val_auc: float` - Best validation AUC score
- `best_epoch: int` - Epoch with best validation

**Methods**:

#### train(num_epochs: int) → void
Main training loop
1. For each epoch:
   - Train on training set
   - Validate on validation set
   - Update learning rate
   - Save checkpoint if improved
   - Check early stopping

#### validate() → dict
Validation evaluation
- Compute predictions on validation set
- Calculate metrics (AUC, F1, etc.)
- Return metrics dictionary

#### save_checkpoint(path: str) → void
Save model checkpoint
- Model state dict
- Optimizer state
- Training metadata

#### load_checkpoint(path: str) → void
Load model checkpoint
- Restore model weights
- Restore optimizer state
- Resume training

**Relationships**:
- **Trains** `StudentModel`
- **Uses** `TeacherModel` (frozen)
- **Uses** `DistillationLoss`
- **Loads from** `ChestXrayDataset`

---

### DistillationLoss
**Purpose**: Combined knowledge distillation + task loss

**File**: `ml/training/kd_losses.py`

**Attributes**:
- `temperature: float` - KD temperature (private, default: 4.0)
- `alpha: float` - Weighting factor (private, default: 0.7)
- `task_loss: BCEWithLogitsLoss` - Binary cross-entropy (private)
- `kd_loss: KLDivLoss` - KL divergence (private)
- `pos_weights: Tensor` - Class balancing weights

**Methods**:

#### forward(student_logits: Tensor, teacher_logits: Tensor, targets: Tensor) → Tensor
Compute total loss
```
L_total = α * L_kd + (1 - α) * L_task
```

#### compute_kd_loss(student: Tensor, teacher: Tensor) → Tensor
Compute distillation loss
- Apply temperature scaling
- Compute KL divergence
- Scale by T²

#### compute_task_loss(logits: Tensor, targets: Tensor) → Tensor
Compute task loss
- Binary cross-entropy with logits
- Apply positive class weights
- Handle class imbalance

**Formula**:
```
L_kd = KL(softmax(z_s/T), softmax(z_t/T)) * T²
L_task = BCE(σ(z_s), y_true, pos_weights)
L_total = α * L_kd + (1 - α) * L_task
```

---

### ChestXrayDataset
**Purpose**: PyTorch dataset for ChestX-ray14

**File**: `ml/data/loader.py`

**Attributes**:
- `df: DataFrame` - Metadata dataframe (private)
- `data_dir: Path` - Image directory (private)
- `transform: Transform` - Data augmentation (private)
- `disease_labels: List[str]` - Disease class names (private)

**Methods**:

#### __len__() → int
Returns dataset size
- Number of images in split

#### __getitem__(idx: int) → Tuple
Get single sample
1. Load image
2. Apply transforms
3. Get labels
4. Return (image, labels) tuple

#### get_labels(idx: int) → Tensor
Parse multi-label annotations
- Convert label string to binary vector
- Handle "No Finding" case

**Usage**:
```python
dataset = ChestXrayDataset(df, data_dir, transform)
loader = DataLoader(dataset, batch_size=32)
```

---

## 5. Configuration

### Config
**Purpose**: Application-wide settings

**File**: `config/config.py`

**Class Attributes** (Static):
- `UPLOAD_DIR: str` - Upload directory path
- `MODEL_PATH: str` - Model checkpoint path
- `ALLOWED_EXTENSIONS: Set[str]` - Valid file types
- `MAX_FILE_SIZE: int` - Maximum upload size (bytes)
- `CONFIDENCE_THRESHOLD: float` - Default threshold (0.5)

**Methods**:

#### get_config() → dict (Static)
Returns configuration dictionary
- For API responses
- For logging

**Usage**:
```python
from config import Config

if file.suffix in Config.ALLOWED_EXTENSIONS:
    # Process file
```

---

### DiseaseLabels
**Purpose**: Disease definitions and utilities

**File**: `config/disease_labels.py`

**Class Attributes** (Static):
- `DISEASE_LABELS: List[str]` - 14 disease names in order

**Methods**:

#### get_risk_level(probability: float) → str (Static)
Map probability to risk level
- Returns "High", "Medium", or "Low"

#### get_risk_color(risk_level: str) → str (Static)
Get hex color for risk level
- High: #c62828 (red)
- Medium: #f9a825 (yellow)
- Low: #2e7d32 (green)

#### get_disease_description(disease: str) → str (Static)
Get medical description of disease

**Constants**:
```python
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", 
    "Infiltration", "Mass", "Nodule", "Pneumonia", 
    "Pneumothorax", "Consolidation", "Edema", 
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]
```

---

## 6. Explainability

### GradCAM
**Purpose**: Gradient-weighted Class Activation Mapping

**File**: `ml/inference/gradcam.py` (or internal to PredictionService)

**Attributes**:
- `model: Module` - Neural network model (private)
- `target_layer: Module` - Layer to visualize (private)
- `gradients: List` - Stored gradients (private)
- `activations: List` - Stored activations (private)

**Methods**:

#### generate_cam(input: Tensor, target_class: int) → ndarray
Generate class activation map
1. Forward pass with hook
2. Backward pass for target class
3. Compute gradient weights
4. Weight activation maps
5. Apply ReLU and normalize
6. Resize to input size

#### compute_gradients(output: Tensor, target: int) → void
Compute gradients for target class
- Backward pass
- Store gradients via hook

#### get_cam_weights() → Tensor
Compute CAM weights
- Global average pooling of gradients
- Returns importance weights

#### overlay_heatmap(image: ndarray, cam: ndarray) → ndarray
Overlay heatmap on original image
- Colorize CAM (jet colormap)
- Alpha blend with image
- Return RGB image

**Algorithm**:
```
1. Forward: x → activations (A)
2. Backward: ∂y_c/∂A → gradients (G)
3. Weights: α_k = GlobalAvgPool(G_k)
4. CAM: ReLU(Σ α_k * A_k)
5. Normalize: CAM / max(CAM)
```

---

## Relationships

### Inheritance (IS-A)
- `TeacherModel` → `Module`
- `StudentModel` → `Module`
- `ConvNeXtBackbone` → `Module`
- `MultiHeadSelfAttention` → `Module`
- `ClassificationHead` → `Module`

### Composition (HAS-A)
- `StudentModel` **has** `ConvNeXtBackbone`
- `StudentModel` **has** `MultiHeadSelfAttention`
- `StudentModel` **has** `ClassificationHead`
- `PredictionResponse` **has** `List[PredictionResult]`

### Association (USES)
- `PredictionService` **uses** `StudentModel`
- `PredictionService` **uses** `GradCAM`
- `PredictionService` **uses** `Config`
- `PredictionService` **uses** `DiseaseLabels`
- `ImageService` **uses** `Config`
- `ReportService` **uses** `Config`
- `KDTrainer` **uses** `StudentModel`
- `KDTrainer` **uses** `TeacherModel`
- `KDTrainer` **uses** `DistillationLoss`

### Dependency (DEPENDS-ON)
- `PredictionService` **creates** `PredictionResponse`
- `KDTrainer` **loads from** `ChestXrayDataset`

---

## Design Patterns

### 1. Service Layer Pattern
**Classes**: `ImageService`, `PredictionService`, `ReportService`

**Purpose**: Separate business logic from API layer

**Benefits**:
- Testability
- Reusability
- Single Responsibility Principle

---

### 2. Singleton Pattern
**Classes**: `Config`, `DiseaseLabels`

**Purpose**: Single instance of configuration

**Implementation**: Class-level attributes and methods

---

### 3. Strategy Pattern
**Classes**: `ConvNeXtBackbone`, `EfficientNetBackbone`, `MobileNetV3Backbone`

**Purpose**: Interchangeable backbone architectures

**Benefits**: Easy to swap implementations

---

### 4. Factory Pattern
**Function**: `create_student_model(model_name: str)`

**Purpose**: Create different model variants

**Example**:
```python
model = create_student_model("convnext_tiny_mhsa")
```

---

### 5. Composition Pattern
**Class**: `StudentModel`

**Purpose**: Compose model from modular components

**Benefits**:
- Flexibility
- Reusability
- Mix-and-match architectures

---

## Object Interactions

### Inference Flow
```
ImageService.load_and_preprocess()
    ↓
PredictionService.predict()
    ↓
StudentModel.forward()
    ↓
    ConvNeXtBackbone.forward()
    ↓
    MultiHeadSelfAttention.forward()
    ↓
    ClassificationHead.forward()
    ↓
GradCAM.generate_cam()
    ↓
PredictionService (format response)
    ↓
PredictionResponse
```

### Training Flow
```
ChestXrayDataset.__getitem__()
    ↓
KDTrainer.train()
    ↓
    TeacherModel.forward() (frozen)
    StudentModel.forward()
    ↓
    DistillationLoss.forward()
    ↓
    Optimizer.step()
    ↓
    KDTrainer.validate()
    ↓
    KDTrainer.save_checkpoint()
```

---

## File Locations

### API Models
- `backend/routes/predict.py` - Request/response models

### Services
- `backend/services/image_service.py` - ImageService
- `backend/services/prediction_service.py` - PredictionService
- `backend/services/report_service.py` - ReportService

### ML Models
- `ml/models/student_model.py` - StudentModel, ConvNeXtBackbone, etc.
- `ml/models/teacher_model.py` - TeacherModel

### Training
- `ml/training/kd_trainer.py` - KDTrainer
- `ml/training/kd_losses.py` - DistillationLoss
- `ml/data/loader.py` - ChestXrayDataset

### Configuration
- `config/config.py` - Config
- `config/disease_labels.py` - DiseaseLabels

### Explainability
- `ml/inference/gradcam.py` or embedded in PredictionService

---

## Key Implementation Details

### Type Annotations
All classes use Python type hints for clarity:
```python
def predict(self, image: Tensor, return_heatmap: bool = True) -> dict:
    ...
```

### Error Handling
Services raise appropriate exceptions:
```python
class ImageService:
    def validate_image(self, file: UploadFile) -> bool:
        if file.size > Config.MAX_FILE_SIZE:
            raise ValueError("File too large")
```

### Lazy Loading
PredictionService loads model on first request:
```python
class PredictionService:
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    def predict(self, ...):
        if not self.model_loaded:
            self.load_model()
```

### Device Agnostic
Models work on CPU or GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

---

## UML Notation Guide

### Visibility
- `+` Public attribute/method
- `-` Private attribute/method
- `$` Static/class attribute/method

### Relationships
- `──>` Association (uses)
- `──|>` Inheritance (is-a)
- `──*` Composition (has-a, strong)
- `──o` Aggregation (has-a, weak)

### Multiplicities
- `1` - Exactly one
- `*` - Zero or more
- `1..*` - One or more

---

## Future Enhancements

### Planned Classes

#### PatientModel
```python
class PatientModel:
    +str patient_id
    +str name
    +int age
    +str gender
    +List[str] history
```

#### DiagnosisRecord
```python
class DiagnosisRecord:
    +str record_id
    +PatientModel patient
    +str image_path
    +PredictionResponse prediction
    +datetime timestamp
```

#### UserModel (Authentication)
```python
class UserModel:
    +str user_id
    +str email
    +str role
    +List[str] permissions
```

---

## Testing

### Unit Tests
Each class should have corresponding test class:
```python
class TestPredictionService:
    def test_predict_returns_dict(self):
        service = PredictionService()
        result = service.predict(test_image)
        assert isinstance(result, dict)
```

### Mocking
Use dependency injection for testability:
```python
class PredictionService:
    def __init__(self, model: StudentModel = None):
        self.model = model or self._load_default_model()
```

---

## References

- UML Class Diagram Notation: https://www.uml-diagrams.org/class-diagrams-overview.html
- Pydantic Models: https://docs.pydantic.dev/
- PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Design Patterns: Gang of Four (GoF) patterns
