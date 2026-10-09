# 🎙️ Grammar Scoring Engine for Spoken Data

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Faster-Whisper](https://img.shields.io/badge/ASR-Faster_Whisper-green)
![RMSE](https://img.shields.io/badge/Training_RMSE-0.8974-brightgreen)

## 📌 Project Overview
This repository contains a complete, end-to-end Machine Learning pipeline designed to automatically evaluate and predict the **Mean Opinion Score (MOS)** for English grammar directly from spoken audio data. 

Instead of relying on black-box deep learning embeddings, this engine prioritizes **high explainability** by transcribing audio into text, engineering targeted linguistic features, and using a gradient-boosted tree to map those features to a human-like Likert score.

## 🏗️ Pipeline Architecture

The pipeline consists of three distinct stages:

### 1. Automatic Speech Recognition (ASR)
* **Model:** `faster-whisper` (Base Model)
* **Compute:** Quantized to `int8` for highly efficient CPU inference.
* **Process:** Converts all spoken audio files into highly accurate text transcripts while dynamically tracking speaking duration to calculate fluency metrics.

### 2. Feature Engineering (NLP)
Once the text is transcribed, we extract highly interpretable linguistic and grammatical features:
* **Fluency:** Words Per Minute (WPM)
* **Grammar Quality:** Absolute grammatical mistakes & syntactic error rate (via `language-tool-python`).
* **Vocabulary Sophistication:** Lexicon count and sentence length.
* **Readability & Complexity:** Flesch Reading Ease, SMOG Index, and Flesch-Kincaid Grade Level (via `textstat`).

### 3. Machine Learning (XGBoost)
* **Algorithm:** `XGBRegressor` (n_estimators=100, max_depth=3, learning_rate=0.05)
* **Validation:** 5-Fold Cross-Validation to ensure robust generalization to unseen audio.
* **Explainability:** Built-in Feature Importance mapping (Weight & Gain) to understand exactly *why* a speaker received a specific score.

## 📊 Evaluation & Results

The model achieved highly stable cross-validation metrics and an excellent final training RMSE:

* **Cross-Validation RMSE:** `1.0822 (± 0.0690)`
* **Cross-Validation Pearson:** `0.4838`
* **Final Training RMSE:** `0.8974`

### 🧠 Key Insights
Based on the XGBoost feature importance analysis, **Flesch Reading Ease** and **Lexicon Count** emerged as the strongest predictive features for grammar scores. This aligns perfectly with human linguistic intuition—speakers who utilize broader vocabularies and construct more complex, sophisticated sentences naturally score higher on the subjective grammar Likert scale.

## 📂 Repository Structure
* `FINAL_PERFECT_NOTEBOOK.ipynb` - The complete Jupyter Notebook containing the ASR implementation, NLP feature extraction, model training, cross-validation, and explainability visualizations.
* `submission.csv` - The final model predictions generated for the unseen test audio dataset.
