# Field Hockey Player Profiling – MSc Dissertation

This repository contains the R scripts used in my MSc Sport Data Analytics dissertation:
**“A Multi-Method Approach to Player Profiling: Integrating Mixed Effects Models, PCA, Clustering, and Discriminant Analysis in Field Hockey.”**

## Methods Used
- Linear Mixed Effects Models (lme4)
- Principal Component Analysis (PCA)
- K-means Clustering
- Linear Discriminant Analysis (LDA)
- Decision Tree Classification
- Feature Engineering of GPS metrics

## Data
GPS data from 397 sessions (matches + training) were analysed to identify:
- Physical dimensions (Workload Volume, Explosiveness, Agility)
- Player archetypes (Balanced, Specialists, Elite Performer)
- Position prediction accuracy using PCA components

## Purpose
This project demonstrates a multi-method analytical pipeline for performance profiling in elite field hockey.

## How to Run
- Install required R packages: `lme4`, `factoextra`, `tidyverse`, `caret`, `rpart`
- Run scripts in the order:
  1. Data Cleaning
  2. Mixed Models
  3. PCA
  4. Clustering
  5. LDA & Decision Trees
