# SatQuery AI

> **Agentic Remote-Sensing Vision-Language Assistant for Satellite Image Understanding**

SatQuery AI is an AI-powered geospatial assistant designed to understand and analyze satellite imagery using **Vision-Language Models (VLMs), remote-sensing data, and agentic AI orchestration**.

The system allows users to interact with satellite images using natural-language queries instead of manually interpreting complex remote-sensing data.

---

## Features

### 1. Remote-Sensing Visual Question Answering

SatQuery AI allows users to ask natural-language questions about satellite imagery.

Example queries:

- What type of land cover is visible?
- Is there an urban settlement in the image?
- Are there water bodies present?
- Is the area predominantly agricultural or urban?
- What objects are visible in the image?

The system uses a fine-tuned **Qwen2.5-VL** model adapted for remote-sensing tasks.

---

### 2. Satellite Image Captioning

The system generates natural-language descriptions of satellite imagery.

It can describe:

- Major land-cover types
- Visible objects
- Spatial arrangements
- Urban areas
- Agricultural areas
- Water bodies
- Overall scene characteristics

Example:

> "The image contains predominantly agricultural land with several rectangular field patterns and scattered vegetation."

---

### 3. Optical-SAR Joint Analysis

SatQuery AI is designed to analyze **co-registered optical and SAR imagery together**.

Optical imagery provides information such as:

- Vegetation
- Land cover
- Visible structures
- Surface appearance

SAR imagery provides complementary information such as:

- Surface characteristics
- Built-up structures
- Surface roughness
- Information under cloudy conditions

The final system will use a dedicated optical-SAR fusion model.

---

### 4. Bi-Temporal Change Understanding

SatQuery AI supports analysis of satellite images acquired at different points in time.

```text
Image at T1
     |
     v
Change Analysis
     |
     v
Image at T2
```

The system is designed to identify changes such as:

- Urban expansion
- Agricultural changes
- Deforestation
- Water-body changes
- Infrastructure development
- Land-cover transitions

The architecture supports both **change description** and **change-oriented Visual Question Answering**.

---

### 5. Agentic AI Controller

SatQuery AI uses an agentic controller to determine which AI capability should handle a user's request.

The controller:

1. Understands the user's query
2. Determines the required task
3. Validates the provided images
4. Checks image count and modality
5. Selects the appropriate specialist model
6. Executes the model
7. Integrates the result
8. Estimates confidence
9. Provides visual evidence
10. Generates an auditable execution summary

### Agent Workflow

```text
                    User Query
                        |
                        v
                +---------------+
                | Agent Router  |
                +-------+-------+
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
         VQA       Captioning      Change
          |             |             |
          |             |             |
          +-------------+-------------+
                        |
                        v
                Specialist Model
                        |
                        v
                Result Integration
                        |
                        v
              Confidence + Evidence
                        |
                        v
                  Final Response
```

---

# Supported Inputs

SatQuery AI is designed to support multiple types of Earth Observation imagery.

| Input Type | Purpose |
|---|---|
| Optical Image | VQA / Captioning |
| Multispectral Image | Remote-sensing analysis |
| SAR Image | SAR analysis |
| Optical + SAR Pair | Cross-modal analysis |
| Bi-Temporal Image Pair | Change understanding |
| GeoTIFF / TIFF | Geospatial imagery |
| PNG / JPEG | Benchmark and general image input |

---

# Technology Stack

## AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA
- Qwen2.5-VL-3B-Instruct
- Vision-Language Models
- Multimodal Fusion Models
- Change Detection Models

## Remote Sensing

- Rasterio
- Sentinel-1 SAR
- Sentinel-2
- Multispectral imagery
- GeoTIFF processing
- Spectral band processing
- Image normalization
- Optical-SAR fusion

## Backend

- Python
- Modular AI agent architecture
- Task Router
- Model Registry
- Image Loader
- Validation Pipeline
- Model Execution Pipeline

## Frontend

- Web-based GUI
- Interactive satellite-image analysis
- Natural-language interaction
- Visual evidence display
- AI-generated responses

---

# Project Structure

```text
SatQueryAI/
|
+-- agent/
|   +-- controller.py
|   +-- loader.py
|   +-- registry.py
|   +-- router.py
|   +-- validator.py
|
+-- data/
|   +-- bigearthnet/
|   |   +-- BigEarthNet-S2/
|   |   +-- processed/
|   |   +-- training_manifest.json
|   |   +-- training_manifest_large.json
|   |
|   +-- samples/
|
+-- models/
|   +-- shared_vlm.py
|   +-- satquery_vlm/
|   |   +-- ...
|   |
|   +-- optical_sar/
|   |   +-- ...
|   |
|   +-- change_detection/
|       +-- ...
|
+-- training/
|   +-- build_large_manifest.py
|   +-- bigearthnet_s2_loader.py
|   +-- vlm_dataset.py
|   +-- vlm_pytorch_dataset.py
|
+-- scripts/
|   +-- test_vlm.py
|
+-- requirements.txt
+-- README.md
+-- ...
```

---

# Dataset

## BigEarthNet

SatQuery AI uses **BigEarthNet** data for remote-sensing model adaptation.

The project uses:

- Sentinel-2 satellite imagery
- Multispectral bands
- Remote-sensing annotations
- VQA examples
- Captioning examples
- Grounding examples

The BigEarthNet text dataset provides training annotations, while the corresponding Sentinel-2 patches provide the actual satellite imagery.

### Dataset Pipeline

```text
BigEarthNet.txt
       |
       v
Annotation Extraction
       |
       v
Training Manifest
       |
       v
Patch Identification
       |
       v
BigEarthNet-S2
       |
       v
Spectral Band Processing
       |
       v
RGB / Model Input
       |
       v
VLM Fine-Tuning
```

---

# VLM Fine-Tuning

The shared Vision-Language Model is based on:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

The model is adapted using **LoRA / PEFT** for remote-sensing tasks.

### Shared VLM Architecture

```text
              Qwen2.5-VL
                   |
                   v
              LoRA Adapter
                   |
                   v
             SatQuery VLM
              /        \
             /          \
            v            v
           VQA       Captioning
```

Using a shared VLM allows SatQuery AI to reuse the same multimodal foundation model for multiple related remote-sensing tasks.

---

# Training Pipeline

```text
Remote-Sensing Dataset
          |
          v
   Annotation Parsing
          |
          v
   Training Manifest
          |
          v
      Image Loader
          |
          v
  Image Normalization
          |
          v
   PyTorch Dataset
          |
          v
      DataLoader
          |
          v
 Qwen2.5-VL + LoRA
          |
          v
 Fine-Tuned Adapter
          |
          v
     SatQuery VLM
```

---

# System Architecture

SatQuery AI follows a **specialist-model architecture** where different remote-sensing tasks can be handled by dedicated models.

```text
                         +----------------+
                         |      User      |
                         +-------+--------+
                                 |
                                 | Natural Language Query
                                 v
                         +---------------+
                         | Agent        |
                         | Controller   |
                         +-------+-------+
                                 |
                                 v
                         +---------------+
                         |  Query Router |
                         +-------+-------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
            v                    v                    v
       +---------+        +-------------+       +-------------+
       |   VQA   |        | Captioning  |       |   Change    |
       |  Model  |        |    Model    |       | Detection  |
       +----+----+        +------+------+       +------+------+
            |                    |                     |
            |                    |                     |
            +--------------------+---------------------+
                                 |
                                 v
                      +---------------------+
                      | Optical-SAR Model   |
                      +----------+----------+
                                 |
                                 v
                       Result Integration
                                 |
                       +---------+---------+
                       |                   |
                       v                   v
                  Confidence           Evidence
                       |                   |
                       +---------+---------+
                                 |
                                 v
                         Final Response
                                 |
                                 v
                         Execution Trace
```

---

# Agent Architecture

The agent layer is divided into multiple components.

## Router

Determines which task is required based on:

- User query
- Number of input images
- Image modality
- Requested operation

Supported tasks include:

```text
vqa
captioning
change_detection
optical_sar
unknown
```

---

## Validator

Validates:

- Image count
- Image format
- Image existence
- Image modality
- Input compatibility
- Required metadata

---

## Loader

Loads supported satellite imagery including:

- GeoTIFF
- TIFF
- PNG
- JPEG

Raster data is processed using **Rasterio**, while standard image formats can be loaded using **PIL**.

---

## Model Registry

The model registry provides a central abstraction for managing specialist AI models.

```text
Task
 |
 +-- VQA
 |
 +-- Captioning
 |
 +-- Change Detection
 |
 +-- Optical-SAR
```

Each task maps to its corresponding model implementation.

---

# Example Queries

## Single Image VQA

```text
What type of land cover is visible in this image?
```

```text
Are there any water bodies visible?
```

```text
Is the area predominantly urban or agricultural?
```

```text
Are roads visible in the image?
```

---

## Image Captioning

```text
Describe this satellite image.
```

```text
Provide a detailed scene description.
```

```text
Describe the major land-cover types and their spatial arrangement.
```

---

## Change Analysis

```text
What changed between these two satellite images?
```

```text
Has urban development increased?
```

```text
Are there any significant land-cover changes?
```

---

## Optical-SAR Analysis

```text
Analyze the optical and SAR images together.
```

```text
What features are visible in both modalities?
```

```text
What additional information does the SAR image provide?
```

---

# Model Execution Flow

```text
User Request
     |
     v
Input Validation
     |
     v
Query Understanding
     |
     v
Task Selection
     |
     v
Model Selection
     |
     v
Model Execution
     |
     v
Output Validation
     |
     v
Confidence Estimation
     |
     v
Evidence Generation
     |
     v
Final Answer
     |
     v
Execution Summary
```

---

# Current Implementation Status

| Component | Status |
|---|---|
| Project Architecture | Completed |
| GUI | Completed |
| Image Preprocessing | Completed |
| Image Validation | Completed |
| Agent Controller | Completed |
| Query Router | Completed |
| Image Loader | Completed |
| BigEarthNet Integration | Completed |
| BigEarthNet-S2 Processing | Completed |
| Training Manifest Generation | Completed |
| PyTorch VLM Dataset | Completed |
| Qwen2.5-VL Integration | Completed |
| Remote-Sensing VLM Fine-Tuning | Completed |
| Shared VLM | Completed |
| VQA | Completed |
| Captioning | Completed |
| Model Registry Integration | In Progress |
| Optical-SAR Model | In Progress |
| Change Detection Model | In Progress |
| Bi-Temporal Change Detection | In Progress |
| Change Description / Change-VQA | In Progress |
| Final End-to-End Integration | Pending |
| Benchmark Evaluation | Pending |
| Deployment Optimization | Pending |

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd SatQueryAI
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the VLM

The shared VLM can be tested using:

```bash
python scripts/test_vlm.py
```

The script loads:

```text
Qwen2.5-VL-3B-Instruct
        +
SatQuery AI LoRA Adapter
```

and performs VQA and captioning inference.

---

# Hardware Considerations

The project uses a multimodal Vision-Language Model and therefore benefits from GPU acceleration.

Recommended:

- NVIDIA GPU
- CUDA-enabled PyTorch
- Sufficient VRAM for multimodal inference
- SSD storage for satellite datasets

The implementation includes quantization and image-processing optimizations to reduce GPU memory requirements.

---

# Design Principles

## Modular

Each specialist model is separated from the agent layer.

## Reusable

The shared VLM can support multiple vision-language tasks.

## Extensible

New models can be added through the model registry without redesigning the complete system.

## Remote-Sensing Aware

The system is specifically adapted for satellite imagery instead of relying only on generic natural-image VLM behavior.

## Auditable

The agent maintains an execution trace containing information such as:

- Selected task
- Selected model
- Input validation
- Model execution
- Output
- Confidence
- Evidence

## Multimodal

The architecture supports:

- Optical imagery
- SAR imagery
- Optical-SAR pairs
- Bi-temporal imagery

---

# Future Work

The remaining development roadmap includes:

1. Complete model registry integration
2. Optical-SAR fusion model
3. Bi-temporal change detection model
4. Change description / change-VQA
5. Change-map generation
6. End-to-end agent integration
7. Confidence calibration
8. Visual evidence generation
9. VRSBench evaluation
10. RSVQA evaluation
11. CDVQA evaluation
12. ISRO/SAC dataset evaluation
13. Deployment optimization

---

# Expected Final Workflow

```text
                         User
                           |
                           v
                 +-------------------+
                 |   SatQuery AI     |
                 |     Interface     |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Input Validation  |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Agent Controller  |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 |  Query Router     |
                 +---------+---------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
       VQA            Captioning         Optical + SAR
        |                  |                  |
        v                  v                  v
   Shared VLM         Shared VLM       Fusion Model
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                    Bi-Temporal Model
                           |
                           v
                 Result Integration
                           |
                  +--------+--------+
                  |                 |
                  v                 v
             Confidence          Evidence
                  |                 |
                  +--------+--------+
                           |
                           v
                    Final Answer
                           |
                           v
                   Execution Trace
```

---

# Project Goal

SatQuery AI aims to make **satellite Earth Observation data accessible through natural-language interaction**.

Instead of requiring users to manually inspect multispectral imagery, SAR data, or temporal image pairs, the system provides an intelligent agent capable of:

- Understanding natural-language queries
- Identifying the required remote-sensing task
- Selecting the appropriate AI model
- Analyzing satellite imagery
- Combining multimodal information
- Understanding temporal changes
- Providing interpretable answers
- Producing visual evidence
- Maintaining an auditable execution trace

The ultimate goal is to build a **unified AI assistant for remote-sensing image understanding, multimodal analysis, and temporal change intelligence**.

---

# Acknowledgements

SatQuery AI builds upon open-source technologies, datasets, and research in Earth Observation and multimodal AI, including:

- BigEarthNet
- Sentinel-1
- Sentinel-2
- Hugging Face Transformers
- Qwen2.5-VL
- PyTorch
- PEFT
- LoRA
- Rasterio

---

# License

This project is intended for academic, research, and hackathon development purposes.

Please check the respective licenses of the underlying datasets, pretrained models, and third-party libraries before redistribution or commercial use.

---

# SatQuery AI

**Ask your satellite imagery. Get an intelligent answer.**
