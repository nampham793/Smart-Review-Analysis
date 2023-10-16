## Smart-Review-Analysis
# Overview:
- Smart Review Analysis is a project that helps businesses and individuals make data-driven decisions by extracting valuable insights from customer reviews and feedback. This model use the reviews of the customers after using the services. After the model have the review, it will return the number of stars for each aspect. In this situation, we have 5 aspects Entertainments, Accomodation Service, Restaurant Service, Food and Beverage, Shopping Services in Vietnamese language.

# Table of contents
  [Technology] <br> 
  [Preprocessing] <br>
  [Installation] <br>
  [Demo]  <br>
  [Train] <br>
  [FastAPI Web Application] <br>

# Technology
  Ensemble <br>
  PhoBert <br>
  VnCoreNLP <br>
  Regex <br>
  KFoldStratified <br>

# Preprocessing
  - Remove emoji
  - Remove special characters
  - Remove punctuation
  - Remove numbers
  - Remove annotations

# Installation
```python
  pip -q install vncorenlp 
  pip -q install iterative-stratification 
  git clone https://github.com/vncorenlp/VnCoreNLP 
  git clone https://github.com/nampham793/Smart-Review-Analysis.git 
```

# Demo
  python app.py

# Train
  Weights: [Model Weight](https://drive.google.com/drive/folders/1SquUNngSVHZAET5mTjw_AT5TtDzjyRBW?usp=sharing)

# FastAPI Web Application
  <p>
    <img width="1439" alt="image" src="https://github.com/nampham793/Smart-Review-Analysis/assets/88274994/500094bc-93f9-4a64-b872-912e60f9f2a0">
</p>
