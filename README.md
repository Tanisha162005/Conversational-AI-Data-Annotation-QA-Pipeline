<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🤖 Conversational AI — Data Annotation & QA Pipeline</h1>

<p align="center">
  <strong>An end-to-end pipeline for annotating customer service conversations, auditing data quality, and training an intent classification model.</strong>
</p>

<p align="center">
  <em>From raw conversations → multi-annotator labeling → quality analysis → compliance audit → ML model</em>
</p>

---

## 🌟 Project Overview

This project implements a **production-grade data annotation and quality assurance pipeline** for conversational AI. It takes raw customer service conversations, applies a structured multi-annotator labeling workflow, validates annotation quality through inter-annotator agreement metrics, performs compliance audits for PII detection, and trains a **Logistic Regression intent classifier** using TF-IDF features.

### ✨ Key Highlights

| Feature | Description |
|---|---|
| 📊 **Multi-Annotator Workflow** | 3 independent annotators + adjudicated ground-truth labels |
| 🏷️ **20 Intent Classes** | Comprehensive taxonomy covering e-commerce customer service scenarios |
| 🔍 **Quality Analysis** | Cohen's Kappa inter-annotator agreement, accuracy metrics & confusion matrices |
| 🛡️ **Compliance Audit** | Automated PII detection (emails, phone numbers, card numbers, URLs) |
| 🤖 **ML Pipeline** | TF-IDF + Logistic Regression with stratified train/validation/test splits |
| 📈 **Model Evaluation** | Accuracy, precision, recall, F1-score & confusion matrix evaluation |
| 🔄 **Continuous Improvement** | Error analysis driving iterative annotation guideline updates |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ANNOTATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Raw Data     │───▶│  Data Annotation  │───▶│  Annotated Data  │   │
│  │  (5000 convs) │    │  (3 Annotators)   │    │  (Adjudicated)   │   │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘   │
│                                                       │             │
│                    ┌──────────────────────────────────┤             │
│                    ▼                                  ▼             │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │    Quality Analysis       │    │      Compliance Audit         │   │
│  │  • Cohen's Kappa          │    │  • PII Detection (Email,     │   │
│  │  • Annotation Accuracy    │    │    Phone, Card, URLs)         │   │
│  │  • Confusion Matrix       │    │  • Duplicate Check            │   │
│  └──────────────────────────┘    │  • Missing Value Audit        │   │
│                                  └──────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                           ML PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Prepare Data  │───▶│  Split Data   │───▶│  Train / Validate     │  │
│  │ (Clean & Map) │    │ (80/10/10)    │    │  / Test               │  │
│  └──────────────┘    └──────────────┘    └───────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Model Architecture                        │    │
│  │     TF-IDF Vectorizer ──▶ Logistic Regression Classifier     │    │
│  │     (unigrams + bigrams)     (max_iter=1000)                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Conversational-AI-Data-Annotation-QA-Pipeline/
│
├── 📁 raw_data/                        # Raw, unannotated conversations
│   ├── conversations.csv               #   → 5,000 synthetic customer service conversations
│   └── data_exploration.py             #   → EDA script (shape, missing values, distributions)
│
├── 📁 intent_taxonomy/                 # Intent classification schema
│   └── intent_taxonomy.csv             #   → 20 intents with definitions & criteria
│
├── 📁 annotation_guidelines/           # Annotator reference material
│   ├── annotation_guidelines.pdf       #   → Rules, examples & edge-case guidance
│   └── guideline_updates.md            #   → Iterative updates post error analysis
│
├── 📁 annotated_data/                  # Multi-annotator labeled dataset
│   └── labeled_conversations.csv       #   → 3 annotator labels + adjudicated intent
│
├── 📁 quality_analysis/                # Annotation quality & error metrics
│   ├── annotation_accuracy.py          #   → Per-annotator accuracy vs. adjudicated label
│   ├── agreement_metrics.py            #   → Pairwise Cohen's Kappa scores
│   ├── confusion_matrix.py             #   → Visual confusion matrix plot
│   ├── error_analysis.py               #   → Analysis of model misclassifications
│   └── model_errors.csv                #   → Output log of detailed model errors
│
├── 📁 compliance_audit/                # Data compliance & PII checks
│   ├── compliance_check.py             #   → Automated 8-point compliance audit
│   └── compliance_report.csv           #   → Audit results log
│
├── 📁 model/                           # ML model training pipeline
│   ├── prepare_data.py                 #   → Clean & prepare features from annotations
│   ├── split_data.py                   #   → Stratified 80/10/10 train/val/test split
│   ├── training.py                     #   → TF-IDF + Logistic Regression training
│   ├── validation.py                   #   → Accuracy & classification report on val set
│   ├── testing.py                      #   → Final evaluation on held-out test set
│   ├── intent_classifier.pkl           #   → Serialized trained model
│   ├── model_dataset.csv               #   → Cleaned ML-ready dataset
│   ├── train.csv                       #   → Training split
│   ├── validation.csv                  #   → Validation split
│   └── test.csv                        #   → Test split
│
├── 📁 model_evaluation/                # Evaluation metric scripts
│   ├── accuracy.py                     #   → Accuracy calculation
│   ├── precision.py                    #   → Precision calculation
│   ├── recall.py                       #   → Recall calculation
│   ├── f1_score.py                     #   → F1-score calculation
│   └── confusion_matrix.py             #   → Model confusion matrix
│
├── 📁 final_report/                    # Deliverables
│   └── MLDA_operations_report.pdf      #   → Final operations report
│
└── requirements.txt                    # Python dependencies
```

---

## 🏷️ Intent Taxonomy (20 Classes)

The taxonomy covers the full spectrum of e-commerce customer service intents:

| ID | Intent | Description |
|---|---|---|
| INT-001 | `order_status` | Status/progress of an existing order |
| INT-002 | `delivery_delay` | Delivery is late or missed |
| INT-003 | `delivery_estimate` | Expected delivery time/date |
| INT-004 | `delivery_tracking` | Shipment location or tracking updates |
| INT-005 | `return_request` | Initiate or inquire about returns |
| INT-006 | `refund_status` | Status of a pending refund |
| INT-007 | `payment_failure` | Payment declined or failed |
| INT-008 | `duplicate_charge` | Charged multiple times for one order |
| INT-009 | `order_cancellation` | Cancel an order before fulfillment |
| INT-010 | `damaged_item` | Product arrived broken/defective |
| INT-011 | `wrong_item` | Received incorrect product |
| INT-012 | `missing_item` | Item absent from delivered package |
| INT-013 | `account_access` | Cannot sign in or access account |
| INT-014 | `password_reset` | Password recovery request |
| INT-015 | `account_security` | Suspicious login or unauthorized access |
| INT-016 | `address_change` | Modify shipping address |
| INT-017 | `subscription_cancel` | Cancel recurring subscription |
| INT-018 | `subscription_billing` | Subscription charge inquiry |
| INT-019 | `promo_code` | Promotional code issue |
| INT-020 | `product_information` | Product features/specs inquiry |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tanisha162005/Conversational-AI-Data-Annotation-QA-Pipeline.git
cd Conversational-AI-Data-Annotation-QA-Pipeline

# 2. Create a virtual environment (recommended)
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 📋 Pipeline Execution Guide

Run each stage sequentially to reproduce the full pipeline:

### Stage 1 — Data Exploration

```bash
python raw_data/data_exploration.py
```
> 🔍 Inspects the raw dataset — shape, missing values, duplicates, channel & language distributions, and message length statistics.

### Stage 2 — Quality Analysis

```bash
python quality_analysis/annotation_accuracy.py
python quality_analysis/agreement_metrics.py
python quality_analysis/confusion_matrix.py
```
> 📊 Evaluates annotation quality: per-annotator accuracy against the adjudicated label, pairwise Cohen's Kappa inter-annotator agreement, and a visual confusion matrix.

### Stage 3 — Compliance Audit

```bash
python compliance_audit/compliance_check.py
```
> 🛡️ Runs an 8-point compliance check: missing values, duplicate IDs, email detection, phone number detection, URL detection, card number detection, synthetic data verification, and overall pass/fail status.

### Stage 4 — ML Dataset Preparation

```bash
python model/prepare_data.py
```
> 🧹 Extracts `customer_message` and `adjudicated_intent` from annotated data, removes nulls and duplicates, and saves the cleaned ML-ready dataset.

### Stage 5 — Data Splitting

```bash
python model/split_data.py
```
> ✂️ Performs a stratified split: **80% train** / **10% validation** / **10% test** — preserving intent distribution across all splits.

### Stage 6 — Model Training

```bash
python model/training.py
```
> 🤖 Trains a **TF-IDF + Logistic Regression** pipeline with unigram & bigram features, English stop word removal, and saves the serialized model as `intent_classifier.pkl`.

### Stage 7 — Validation

```bash
python model/validation.py
```
> ✅ Evaluates the trained model on the validation set — reports overall accuracy and a detailed per-intent classification report (precision, recall, F1-score).

### Stage 8 — Model Testing & Evaluation

```bash
python model/testing.py
python model_evaluation/accuracy.py
python model_evaluation/precision.py
python model_evaluation/recall.py
python model_evaluation/f1_score.py
python model_evaluation/confusion_matrix.py
```
> 🎯 Runs the final test on the held-out test split, calculating discrete metrics (accuracy, precision, recall, F1) and generating `model_evaluation/model_confusion_matrix.png`.

### Stage 9 — Error Analysis

```bash
python quality_analysis/error_analysis.py
```
> 🔍 Analyzes model misclassifications on the test set, outputting `quality_analysis/model_errors.csv`. Findings from this analysis drive iterative updates in `annotation_guidelines/guideline_updates.md`.

---

## 🧠 Model Details

| Component | Configuration |
|---|---|
| **Vectorizer** | TF-IDF (unigrams + bigrams, lowercase, English stop words) |
| **Classifier** | Logistic Regression (max_iter=1000) |
| **Data Split** | 80% Train / 10% Validation / 10% Test (stratified) |
| **Input** | Raw `customer_message` text |
| **Output** | Predicted intent label (20 classes) |
| **Serialization** | `joblib` → `model/intent_classifier.pkl` |

---

## 🔬 Quality Assurance Framework

### Annotation Quality

| Metric | What It Measures |
|---|---|
| **Cohen's Kappa** | Pairwise inter-annotator agreement (chance-corrected) |
| **Annotation Accuracy** | Each annotator's agreement with the adjudicated ground truth |
| **Confusion Matrix** | Visual map of annotation disagreements by intent |
| **Error Analysis** | Deep-dive into model misclassifications to refine guidelines |

### Compliance Audit (8-Point Check)

| # | Check | Description |
|---|---|---|
| 1 | Missing Values | Required fields completeness |
| 2 | Duplicate IDs | Unique conversation identifiers |
| 3 | Email Detection | PII — email address patterns |
| 4 | Phone Detection | PII — phone number patterns |
| 5 | URL Detection | External URL references |
| 6 | Card Detection | PII — credit/debit card-like sequences |
| 7 | Synthetic Flag | All records marked as synthetic data |
| 8 | Overall Status | Aggregate PASS / REVIEW REQUIRED |

---

## 📊 Data Schema

### Raw Data (`conversations.csv`)

| Column | Type | Description |
|---|---|---|
| `conversation_id` | string | Unique conversation identifier (CONV-XXXXX) |
| `channel` | string | Communication channel (email, web_form, etc.) |
| `customer_message` | string | Raw customer message text |
| `language` | string | Message language code |
| `synthetic` | boolean | Whether the data is synthetically generated |

### Annotated Data (`labeled_conversations.csv`)

| Column | Type | Description |
|---|---|---|
| `conversation_id` | string | Unique conversation identifier |
| `channel` | string | Communication channel |
| `customer_message` | string | Raw customer message text |
| `language` | string | Message language code |
| `synthetic` | boolean | Synthetic data flag |
| `annotator_1_label` | string | Annotator 1's intent label |
| `annotator_2_label` | string | Annotator 2's intent label |
| `annotator_3_label` | string | Annotator 3's intent label |
| `adjudicated_intent` | string | Final resolved intent label |
| `annotation_confidence` | string | Confidence level (high/medium/low) |
| `review_status` | string | Review status (approved/flagged) |
| `reviewer_id` | string | Reviewer identifier |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **pandas** | Data manipulation & analysis |
| **scikit-learn** | ML pipeline, metrics & model training |
| **matplotlib** | Visualization (confusion matrices) |
| **seaborn** | Statistical data visualization |
| **numpy** | Numerical computations |
| **joblib** | Model serialization |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with ❤️ for Conversational AI</strong>
</p>
