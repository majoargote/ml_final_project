# Product Requirements Document
## MIMIC-III Hospital Mortality Classification Pipeline

**Student:** Mariajose Argote  
**Course:** Computational Machine Learning  
**Deadline:** December 11, 2025 @ 08:00 (notebook) / 18:00 (predictions)  
**Presentation:** December 12, 2025 @ 10:00

---

## 1. Project Overview

### Objective
Build a classification pipeline to predict `HOSPITAL_EXPIRE_FLAG` (patient mortality during ICU stay) using the MIMIC-III dataset subset with 20,885 ICU patient observations.

### Success Criteria
- Working, reproducible Jupyter notebook with complete ML pipeline
- Predictions submitted as probabilities (.predict_proba) in CSV format
- Top-tier prediction ranking (within reasonable distance of best performer)
- Ability to defend all modeling decisions during in-person presentation
- Grade target: 9/10 from presentation + up to 1/10 from prediction ranking

---

## 2. Data Specifications

### Dataset Characteristics
- **Observations:** 20,885 ICU stays
- **Features:** 44 total variables (need to identify and drop 3 giveaway variables)
- **Target:** `HOSPITAL_EXPIRE_FLAG` (binary: 0=survived, 1=died)

### Variables to Identify and Remove
**Critical:** These are "giveaways" that leak target information:
- `DOD` (Date of Death) - directly reveals mortality
- `DEATHTIME` (Death datetime) - directly reveals mortality  
- `LOS` (Length of Stay) - this is the OTHER target variable for regression task

### Feature Categories

**ID Columns (3):** Keep for tracking, drop before modeling
- `subject_id`, `hadm_id`, `icustay_id`

**Vital Signs (21 features):** Min/Max/Mean for each
- Heart Rate, Systolic BP, Diastolic BP, Mean BP, Respiration Rate, Temperature, SpO2, Glucose

**Temporal Features (4):**
- `DOB`, `ADMITTIME`, `DISCHTIME`, `Diff`

**Demographics (5):**
- `GENDER`, `ADMISSION_TYPE`, `INSURANCE`, `RELIGION`, `MARITAL_STATUS`, `ETHNICITY`

**Clinical (3):**
- `DIAGNOSIS`, `ICD9_diagnosis`, `FIRST_CAREUNIT`

**Final feature count:** 39 explanatory variables (after dropping 3 giveaways + target)

---

## 3. Pipeline Architecture

### Phase 1: Exploratory Data Analysis (EDA)
**Goal:** Understand data quality and patterns

- Load train/test datasets
- Check data types, missing values, distributions
- Identify and visualize class imbalance in target
- Examine correlations between vital signs
- Analyze temporal patterns (age at admission, time-based features)
- Explore categorical variable distributions
- **Analyze external diagnoses data:**
  - Check distribution of diagnoses per patient
  - Identify most common primary diagnoses
  - Examine relationship between diagnosis count and mortality
- Document findings in markdown cells

### Phase 2: Data Preprocessing & Feature Engineering
**Goal:** Prepare clean, model-ready features with external data integration

**2.1 Feature Removal**
- Drop ID columns: `subject_id`, `hadm_id`, `icustay_id`
- Drop leakage features: `DOD`, `DEATHTIME`, `LOS`

**2.2 External Data Integration: Diagnoses Features**
**Critical:** All diagnosis severity calculations use ONLY training data to avoid leakage

**Step 1: Volume Features**
```python
# Count total diagnoses per patient
diag_counts = diagnoses.groupby('HADM_ID').size().reset_index(name='diagnosis_count')
```

**Step 2: Primary Diagnosis Features**
```python
# Extract primary diagnosis (SEQ_NUM = 1)
primary_diag = diagnoses[diagnoses['SEQ_NUM'] == 1][['HADM_ID', 'ICD9_CODE']]
# One-hot encode top N most common primary diagnoses
# Decision: Start with N=20, validate with cross-validation
```

**Step 3: Severity Scoring with Bayesian Smoothing**
```python
# Calculate mortality rate per ICD9 code (TRAINING DATA ONLY)
severity_scores = train_diagnoses.groupby('ICD9_CODE').agg({
    'HOSPITAL_EXPIRE_FLAG': ['sum', 'count']
}).reset_index()

# Apply Bayesian smoothing to handle rare diagnoses
overall_mortality = train['HOSPITAL_EXPIRE_FLAG'].mean()
alpha = 10  # Confidence parameter (tune via cross-validation)

severity_scores['smoothed_mortality'] = (
    (severity_scores['count'] * severity_scores['mean'] + alpha * overall_mortality) / 
    (severity_scores['count'] + alpha)
)
```

**Step 4: Aggregate Severity Metrics**
```python
# Merge severity back to all diagnoses, then aggregate per patient:
# - avg_diagnosis_severity: Mean of all diagnosis severity scores
# - max_diagnosis_severity: Highest severity score
# - primary_diagnosis_severity: Severity of primary diagnosis only
```

**Step 5: Handle Missing Diagnoses**
```python
# Patients without diagnoses → NaN values
# Strategy: Fill with defaults (not drop!)
df['diagnosis_count'].fillna(0, inplace=True)
df['avg_diagnosis_severity'].fillna(overall_mortality, inplace=True)
df['max_diagnosis_severity'].fillna(overall_mortality, inplace=True)
df['primary_diagnosis_severity'].fillna(overall_mortality, inplace=True)
```

**Justification for approach:**
- Bayesian smoothing balances observed rates with prior knowledge
- Automatically handles rare diagnoses by shrinking toward overall rate
- No arbitrary thresholds needed for minimum sample size
- Preserves information from all diagnoses while being statistically sound

**2.3 Additional Feature Engineering (Main Dataset)**
- Create age at admission: `ADMITTIME - DOB`
- Consider: admission hour, day of week, season
- Consider: vital sign ranges (Max - Min)
- Consider: stability indicators (coefficient of variation)

**2.4 Missing Value Handling**
- Document missing value patterns
- Choose strategy: imputation (mean/median/mode) or indicators
- Justify approach in markdown

**2.5 Categorical Encoding**
- One-hot encoding for nominal variables (insurance, ethnicity, etc.)
- Consider target encoding for high-cardinality features
- Handle rare categories appropriately

**2.6 Scaling**
- Standardize/normalize numerical features
- **Critical:** Save scaler parameters for test set transformation
- Document which features were scaled

### Phase 3: Model Development
**Goal:** Build and tune classification models

**3.1 Train/Validation Split**
- Stratified split (maintain class balance)
- Consider time-based split if temporal patterns exist
- Typical split: 80/20 or use cross-validation

**3.2 Baseline Models**
Test multiple algorithm families:
- Logistic Regression (interpretable baseline)
- Random Forest (handles non-linearity, feature importance)
- Gradient Boosting (XGBoost/LightGBM - high performance)
- Support Vector Machine (optional)
- Neural Network (optional, if time permits)

**3.3 Handling Class Imbalance**
Likely issue with mortality prediction:
- Check class distribution
- Apply techniques: class_weight='balanced', SMOTE, under/oversampling
- Evaluate impact on model performance

**3.4 Hyperparameter Tuning**
- Use GridSearchCV or RandomizedSearchCV
- Document parameter spaces explored
- Justify final hyperparameter choices
- **Key parameters to tune:**
  - Tree-based: max_depth, n_estimators, min_samples_split
  - Regularization: C (LogReg), alpha (Ridge/Lasso)
  - Learning rate, subsample (boosting)

**3.5 Model Evaluation**
- Metrics to track: ROC-AUC, PR-AUC, F1, recall, precision
- Confusion matrix analysis
- Feature importance plots
- Cross-validation scores
- **Emphasize:** Predictions will be scored on undisclosed metric
  - Focus on probability calibration
  - Avoid overfitting (prioritize generalization)

### Phase 4: Final Model Selection
**Goal:** Choose best model for test predictions

**Selection Criteria:**
- Best cross-validation performance
- Reasonable training time
- Interpretability vs. performance trade-off
- Robustness across validation folds

**Model Ensemble (Optional):**
- Consider averaging predictions from top 2-3 models
- Voting classifier or stacking

### Phase 5: Test Predictions
**Goal:** Generate submission file

**Critical Steps:**
1. Apply EXACT same preprocessing to test set
2. Use saved scalers, encoders from training
3. Generate probability predictions (`.predict_proba`)
4. Format as required: two columns if needed, check sample submission
5. Verify no NaN values, correct shape
6. Save as: `mirabent_guillem_CML_2025.csv`

---

## 4. Code Organization Requirements

### Notebook Structure
Use markdown sections with clear headings:

```
1. Project Setup & Imports
2. Data Loading & Overview
3. Exploratory Data Analysis
4. Data Preprocessing
   4.1 Feature Engineering
   4.2 Missing Values
   4.3 Encoding
   4.4 Scaling
5. Model Development
   5.1 Baseline Models
   5.2 Hyperparameter Tuning
   5.3 Model Evaluation
6. Final Model Selection
7. Test Set Predictions
8. Conclusions & Next Steps
```

### Code Quality Standards
- Clean, readable code with meaningful variable names
- Comments explaining non-obvious logic
- Modular functions for repeated operations
- No unused code blocks
- Warning about non-standard packages if used

### Documentation Requirements
Every major decision needs justification:
- "Why did you drop these features?"
- "Why this imputation strategy?"
- "Why these hyperparameters?"
- "Why this final model?"

Acceptable justifications:
✅ "Best cross-validation performance after GridSearch"
✅ "Domain knowledge suggests mortality correlates with X"
✅ "Handling class imbalance improved recall by 15%"

❌ "Just felt right"
❌ "Tutorial said so"

---

## 5. Presentation Preparation

### Questions to Prepare For
1. **Pipeline decisions:** Why each preprocessing step?
2. **Model choice:** Why this algorithm over alternatives?
3. **Hyperparameters:** How did you select these values?
4. **Evaluation:** What metrics did you optimize for and why?
5. **Code-level:** What does line X do? Why this approach?
6. **Generalization:** How did you prevent overfitting?
7. **Feature engineering:** Rationale for created features?
8. **Class imbalance:** How did you handle it?

### Defense Strategy
- Know your code inside-out (every line)
- Have evidence for every claim ("as shown in cell X...")
- Admit uncertainties honestly ("I tried X and Y; X performed better")
- Discuss trade-offs (interpretability vs. accuracy, speed vs. performance)

---

## 6. Deliverables Checklist

### Before December 11 @ 08:00
- [ ] Complete Jupyter notebook: `mirabent_guillem_CML_2025.ipynb`
- [ ] All code cells run top-to-bottom without errors
- [ ] Markdown documentation in place
- [ ] Code produces submitted predictions
- [ ] Notebook uploaded to submission system

### Before December 11 @ 18:00
- [ ] Test predictions file: `mirabent_guillem_CML_2025.csv`
- [ ] Predictions are probabilities (from .predict_proba)
- [ ] Correct format matching sample submission
- [ ] File uploaded to Kaggle competition

### Before December 12 @ 10:00
- [ ] Review notebook thoroughly
- [ ] Practice explaining key decisions
- [ ] Prepare to answer technical questions
- [ ] Test running notebook from scratch

---

## 7. Risk Mitigation

### Common Pitfalls to Avoid
1. **Data leakage:** Verify DOD, DEATHTIME, LOS are removed
2. **Test set contamination:** Never fit scalers/encoders on test data
3. **Overfitting:** Use cross-validation, don't overtune on validation set
4. **Wrong prediction format:** Submit probabilities, not classes
5. **Code doesn't run:** Test full notebook execution before submission
6. **Missing justifications:** Document reasons for every major choice
7. **Non-standard packages:** Warn about special dependencies

### Time Management
- **Week 1 (Dec 2-5):** EDA + preprocessing
- **Week 2 (Dec 6-9):** Modeling + tuning
- **Dec 10:** Final model + test predictions
- **Dec 11 morning:** Buffer for issues, final submission
- **Dec 11 evening:** Presentation prep
- **Dec 12:** Presentation

---

## 8. Grading Breakdown

| Component | Points | Requirements |
|-----------|--------|--------------|
| **Working Code** | Part of 9 | Runs correctly, generates predictions |
| **Understanding** | Part of 9 | Justify all decisions, explain every line |
| **Presentation** | **9 total** | Answer questions with methodological justification |
| **Prediction Ranking** | **0-1** | Based on undisclosed metric vs. class performance |
| **Total** | **10** | Max grade (can't exceed even with both tasks) |

---

## 9. Success Metrics

**Minimum Viable Product:**
- Notebook runs end-to-end
- Predictions submitted on time
- Can explain all code and decisions
- **Target: 7-8/10**

**Target Product:**
- Clean, well-documented pipeline
- Multiple models evaluated with GridSearch
- Top 25% of prediction ranking
- Confident presentation defense
- **Target: 9-10/10**

**Stretch Goals:**
- Ensemble methods
- Advanced feature engineering
- Top 3 prediction ranking
- Creative visualization
- **Potential: Full 10/10 + recognition**