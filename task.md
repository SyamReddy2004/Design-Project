# AI-Based Grading System for Tender Coconut & Turmeric

## Overview
Build a web application to grade Tender Coconuts and Turmeric based on images using AI.

## Requirements (Assumed)
- **User Interface**: Modern web interface to upload images.
- **Backend API**: Process images and return grading.
- **Grading Criteria**:
  - **Tender Coconut**: Size, Shape, Color (Green freshness), Defects.
  - **Turmeric**: Color (Curcumin content indicator), Size/Shape of rhizomes, Surface texture.
- **AI Model**: Deep Learning model (CNN) to classify/regress grade.

## Implementation Plan

### Phase 1: Project Setup & UI Foundation
- [ ] Initialize Project Structure (Frontend: Vite/React, Backend: FastAPI/Python).
- [ ] Setup Basic UI with Upload Component.
- [ ] Setup Backend API skeleton.

### Phase 2: Backend Logic & Mock Model
- [ ] Implement Image Upload Endpoint.
- [ ] Create a "Mock" AI Grader (returns random/heuristic grades for demo).
- [ ] Integration: Connect Frontend to Backend.

### Phase 3: Advanced Features (To be discussed)
- [ ] **Data Collection**: Mechanism to save uploaded images for training.
- [ ] **Real Model**: Train a TensorFlow/PyTorch model (Requires Dataset).
- [ ] **Dashboard**: View history of graded items.

## Questions for User
1. Do you have a dataset of Coconut/Turmeric images with grades?
2. What are the specific grading parameters (e.g., Grade A, B, C or Score 1-100)?
3. Should this run locally or be deployed?
