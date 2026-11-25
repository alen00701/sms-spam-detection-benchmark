# Comparative Benchmarking for SMS Spam Detection

A benchmark of classical text classifiers for SMS spam detection: **Naive Bayes, Logistic Regression and Linear SVM** on TF-IDF features. It also compares three ways to handle class imbalance, with and without hyperparameter tuning. The best models are combined into a weighted-voting ensemble.

> Machine Learning course project, Department of Computer Science, School of Engineering and Digital Sciences, Nazarbayev University.
> Authors: Alen Kazhymukan, Nurlan Boranbay, Darkhan Shagambayev. Supervisor: Hari Mohan Rai, Ph.D.
> 📄 [Read the report](paper.pdf)

## Results

Hold-out test set: 1,115 messages (966 ham, 149 spam).

| Model | Strategy | Accuracy | Precision | Recall | F1 (spam) |
|---|---|---:|---:|---:|---:|
| **Linear SVM** | **Baseline, tuned** | **0.9883** | **1.0000** | **0.9128** | **0.9544** |
| Linear SVM | Oversampled | 0.9874 | 0.9856 | 0.9195 | 0.9514 |
| Linear SVM | Class weight | 0.9865 | 0.9786 | 0.9195 | 0.9481 |
| Naive Bayes | Baseline, tuned | 0.9857 | 0.9854 | 0.9060 | 0.9441 |
| Logistic Regression | Oversampled, tuned | 0.9848 | 0.9714 | 0.9128 | 0.9412 |
| Naive Bayes | Baseline | 0.9776 | 0.9921 | 0.8389 | 0.9091 |
| Logistic Regression | Baseline | 0.9731 | 0.9837 | 0.8121 | 0.8897 |

**Ensemble** (tuned Linear SVM + tuned Naive Bayes + oversampled Logistic Regression, weighted voting): accuracy **98.83%**, F1 0.9544, MCC 0.949, and **no legitimate message marked as spam** (precision 1.00).

Because the single models were already at about 0.95 F1, the ensemble did not improve the metrics much. Most remaining errors are spam messages that slip through, which is the safer kind of mistake than losing real messages.

## Method

- **Preprocessing:** lowercasing, keeping letters only, removing English stopwords (NLTK).
- **Features:** TF-IDF with unigrams and bigrams.
- **Imbalance strategies:** baseline, class weights, random oversampling (imbalanced-learn).
- **Tuning:** `GridSearchCV` with 5-fold stratified cross-validation, optimising F1.
- **Interpretation:** the Linear SVM's largest weights show typical spam words (*mobile, txt, claim, prize, ringtone*) and typical personal-message words (*remember, told, yup, home, sleep*).

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,572 English SMS messages (4,825 ham, 747 spam), included in `data/spam.csv`.
Licensed under CC BY 4.0 by T. Almeida and J. M. Gómez Hidalgo:

> Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* ACM Symposium on Document Engineering.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook sms_spam_benchmark.ipynb
```

## Repository structure

```
sms_spam_benchmark.ipynb   preprocessing, benchmarking, tuning, ensemble, plots
data/spam.csv              SMS Spam Collection dataset
paper.pdf                  project report
requirements.txt
```
