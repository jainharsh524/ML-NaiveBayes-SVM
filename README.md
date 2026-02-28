
# Machine Learning Classification: Naive Bayes vs Support Vector Machines

Author: Harsh Jain  
Course: Machine Learning  

---

## Project Summary

This project presents a structured comparative study of two supervised learning algorithms:

- Naive Bayes (probabilistic generative model)
- Support Vector Machine (margin-based discriminative model)

The goal was to evaluate model behavior under different hyperparameter configurations, analyze bias-variance tradeoffs, and study the effect of feature assumptions on real datasets.

The project emphasizes reproducible experimentation, cross-validation rigor, and analytical interpretation of results rather than treating models as black boxes.

---

## Business Relevance

Classification models are widely used in:

- Medical diagnosis prediction
- Spam filtering
- Risk scoring
- Fraud detection
- Customer segmentation

This project demonstrates the ability to:

- Design controlled ML experiments
- Tune hyperparameters systematically
- Interpret performance metrics correctly
- Diagnose overfitting and underfitting
- Understand theoretical assumptions and their real-world impact

---

## Repository Structure

```

ML-NaiveBayes-SVM/
│
├── reports/
│   ├── Naive_Bayes_Report.pdf
│   └── SVM_Report.pdf
│
├── notebooks/
│   ├── Naive_Bayes.ipynb
│   └── SVM.ipynb
│
├── data/
│   ├── diabetes.csv
│   └── mushroom/
│
└── README.md

```

---

## 1. Naive Bayes Implementation

### Key Components

- Implemented with Laplace smoothing
- Explored multiple smoothing values (α)
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

### Experimental Focus

- Impact of smoothing on generalization
- Bias-variance tradeoff analysis
- Effect of correlated (duplicate) features
- Demonstrated how independence assumption violations degrade test performance

### Technical Insight

Duplicate features artificially amplify likelihood terms:

\[
P(y) \prod_i P(x_i | y)
\]

When a feature is repeated, its likelihood is exponentiated, causing overconfidence and reduced generalization. This experiment validates theoretical limitations of Naive Bayes in practice.

---

## 2. Support Vector Machine Implementation

### Methodology

- Implemented using scikit-learn
- Applied feature standardization
- 5-fold Stratified Cross-Validation
- Exhaustive grid search across kernels and hyperparameters

### Kernels Evaluated

- Linear
- Polynomial
- RBF (Radial Basis Function)

### Hyperparameters Tuned

- C (regularization strength)
- Gamma (kernel coefficient)
- Degree (polynomial kernel)

### Observations

- RBF kernel achieved best validation performance
- Moderate regularization provided optimal bias-variance balance
- High C values increased risk of overfitting
- Feature scaling significantly improved performance

This demonstrates understanding of geometric margin theory and kernel transformations.

---

## Evaluation Strategy

- Stratified K-Fold Cross Validation (k = 5)
- Mean accuracy across folds
- Reproducibility ensured via fixed random state
- No data leakage during training

The evaluation pipeline mirrors production-grade experimentation standards.

---

## Tools and Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

Core ML concepts applied:

- Generative vs Discriminative models
- Bias-Variance tradeoff
- Regularization
- Kernel methods
- Feature scaling
- Cross-validation
- Overfitting diagnostics

---

## Key Results

- Optimal Naive Bayes smoothing achieved strong generalization performance
- Demonstrated measurable degradation when independence assumption is violated
- RBF SVM outperformed linear and polynomial kernels
- Identified regularization sweet spot for optimal generalization

---

## What This Project Demonstrates

- Strong theoretical grounding in machine learning fundamentals
- Ability to implement algorithms from scratch and via libraries
- Experimental design skills
- Hyperparameter optimization capability
- Analytical thinking beyond surface-level metrics
- Clean project structuring and reproducibility practices

---

## How to Run

Clone the repository:

```

git clone [https://github.com/yourusername/ML-NaiveBayes-SVM.git](https://github.com/yourusername/ML-NaiveBayes-SVM.git)
cd ML-NaiveBayes-SVM

```

Launch Jupyter Notebook:

```

jupyter notebook

```

Run notebooks inside the `notebooks/` directory.

---

## Future Improvements

- Automated hyperparameter optimization (Bayesian search)
- Model comparison with Logistic Regression and Random Forest
- ROC-AUC and calibration analysis
- Dimensionality reduction with PCA before SVM
- Deployment-ready model packaging

---

## Conclusion

This project provides a rigorous comparative analysis of probabilistic and margin-based classifiers. It demonstrates both theoretical understanding and practical implementation skills required for real-world machine learning tasks.

The experiments highlight how assumptions, regularization, and kernel choice directly influence model generalization performance.
