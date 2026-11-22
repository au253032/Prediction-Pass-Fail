# 🎓 Student Pass/Fail Prediction Using a Decision Tree

<img width="3780" height="1971" alt="decision_tree" src="https://github.com/user-attachments/assets/1a18d85b-3641-4d5d-9fff-780908a9b729" />
<img width="1544" height="1176" alt="confusion_matrix (1)" src="https://github.com/user-attachments/assets/42ea28b2-48e1-4a11-904e-643b14741de0" />




A beginner-friendly machine learning project that predicts whether a student will **pass or fail** from three inputs: **study hours**, **attendance** and **previous score**. It walks through the full workflow in one notebook: building the dataset, training a Decision Tree classifier, evaluating it, visualizing the tree, predicting for a new student, and saving the model.

🔗 **Run it on Kaggle:** [prediction-pass-fail](https://www.kaggle.com/code/malikyasir23/prediction-pass-fail)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Model Configuration](#-model-configuration)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Example Prediction](#-example-prediction)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Limitations](#%EF%B8%8F-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🔎 Overview
Spotting students who are at risk of failing early lets teachers and institutions help them sooner. This project frames the problem as **binary classification** (`1` = Pass, `0` = Fail) and uses a **Decision Tree**, a model that is easy to read and explain because every prediction follows a clear chain of if/else rules.

---

## 📊 Dataset
The notebook creates a small **synthetic dataset of 20 students**. The data is written directly in the code and is also exported to `student_dataset.csv`.

| Feature | Description |
|---|---|
| `study_hours` | Hours the student studies |
| `attendance` | Attendance percentage |
| `previous_score` | Score in a previous assessment |
| `result` | **Target:** `1` = Pass, `0` = Fail |

- **Shape:** 20 rows × 4 columns
- **Missing values:** none
- **Class balance:** 10 Fail, 10 Pass (perfectly balanced)

---

## 🧪 Workflow
1. Create and inspect the dataset (shape, data types, missing values, class counts)
2. Separate the features (`X`) from the target (`y`)
3. Split into training and test sets (80/20, stratified, `random_state=42`)
4. Train a `DecisionTreeClassifier`
5. Evaluate with accuracy, a confusion matrix and a classification report
6. Visualize the confusion matrix and the decision tree
7. Predict the result and class probabilities for a new student
8. Save the model, dataset, predictions and charts

---

## ⚙️ Model Configuration
| Setting | Value |
|---|---|
| Algorithm | `DecisionTreeClassifier` (scikit-learn) |
| Criterion | `gini` |
| Max depth | `3` |
| Train/test split | 80% / 20% (16 train, 4 test), stratified |
| Random state | `42` |

---

## 📈 Results
| Metric | Fail | Pass | Overall |
|---|---|---|---|
| Precision | 1.00 | 1.00 | |
| Recall | 1.00 | 1.00 | |
| F1-score | 1.00 | 1.00 | |
| Accuracy | | | **100%** (4 of 4 test samples) |

**Confusion matrix**

|  | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 2 | 0 |
| **Actual Pass** | 0 | 2 |

> ⚠️ These scores come from only **4 test samples** on a small synthetic dataset, so they show that the pipeline works, **not** how well the model would perform on real students. See [Limitations](#%EF%B8%8F-limitations).

---

## 🖼 Visualizations
| Confusion Matrix | Decision Tree |
|---|---|
| ![Confusion Matrix](images/confusion_matrix.png) | ![Decision Tree](images/decision_tree.png) |

---

## 🔮 Example Prediction
For a new student with **6 study hours, 80% attendance and a previous score of 75**:

```
Predicted Result: PASS
Probability of Fail: 0.0 %
Probability of Pass: 100.0 %
```

---

## 💻 Installation
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the notebook:
```bash
jupyter notebook Decision_Tree_Classification.ipynb
```
Or open it directly on [Kaggle](https://www.kaggle.com/code/malikyasir23/prediction-pass-fail).

To reuse the saved model in your own code:
```python
import joblib
import pandas as pd

model = joblib.load("decision_tree_model.pkl")

student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [80],
    "previous_score": [75],
})

print("PASS" if model.predict(student)[0] == 1 else "FAIL")
```
> Only load `.pkl` files from sources you trust, because pickle files can run code when loaded.

---

## 📁 Project Structure
```
├── Decision_Tree_Classification.ipynb  # Main notebook
├── student_dataset.csv               # Generated dataset (20 students)
├── decision_tree_model.pkl           # Trained model (joblib)
├── decision_tree_predictions.csv     # Actual vs predicted on the test set
├── images/
│   ├── confusion_matrix.png
│   └── decision_tree.png
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib
- **Environment:** Jupyter Notebook / Kaggle

---

## ⚠️ Limitations
- The dataset is **synthetic and tiny** (20 rows), so it does not reflect real student behavior.
- The test set has only **4 samples**. A perfect score on 4 samples is not strong evidence of a good model.
- A decision tree trained on only 16 rows can easily **overfit**, even with a limited depth.
- No cross-validation or hyperparameter tuning has been done yet.

---

## 🚀 Future Improvements
- Use a larger, real-world student performance dataset
- Evaluate with k-fold cross-validation instead of a single split
- Tune hyperparameters (`max_depth`, `min_samples_leaf`) with `GridSearchCV`
- Compare with Logistic Regression and Random Forest
- Add feature importance analysis
- Build a small Streamlit app so anyone can enter a student's data and get a prediction

---

## 👤 Author
**Ateeq Ur Rehman** · BS Electronics Student | Machine Learning Learner
- GitHub: `[https://github.com/au253032]`

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
