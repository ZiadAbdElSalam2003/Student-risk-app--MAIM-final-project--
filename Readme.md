# 🎓 Student Risk App

A prototype web application that helps schools and teachers detect and address student risks early.

The app predicts **student risk levels (High / Medium / Low)** in **Portuguese** and **Math** using ML models, and performs **feedback sentiment analysis** to understand opinions from students and parents.

The interface is built with **Flask** for simple, user-friendly access by teachers and school administrators.

---

## ✨ Key Features

* **Risk Prediction:**

  * Portuguese: Random Forest
  * Math: Decision Tree
* **Sentiment Analysis:** Pre-trained NLP model for analyzing student/parent feedback.
* **Flask Web Demo:** Easy-to-use prototype for non-technical users.
* **End-to-End Pipeline:** From data cleaning & preprocessing → model training & evaluation → deployment in a demo app.

---

## 🎥 Demo

[![Watch the demo](https://www.canva.com/design/DAG0qsgCGx8/HxRH0zkTqnmTb65F2ByYXA/edit?utm_content=DAG0qsgCGx8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)](https://youtu.be/4Kk59-epa2M)



---

## 📑 Technical Report

For detailed methodology, data preprocessing steps, feature engineering, model evaluation, and explanation of the full project workflow, please check the **report.pdf** included in this repository.

---

## 🚀 Value & Impact

* Enables **early detection** of students at risk of failure, so schools can take **proactive measures**.
* Provides **AI-driven sentiment insights** to capture opinions from students and parents.
* Helps educators and administrators better understand student challenges and improve overall learning outcomes.

---

## ⚙️ Tech Stack

* **Python** (pandas, scikit-learn)
* **Flask** (web interface)
* **Machine Learning** (Decision Tree, Random Forest)
* **NLP** (pre-trained sentiment model)

---

## 📂 Project Structure

```
student-risk-app/
│
├── app.py              # Flask app
├── models/             # Saved trained models
├── static/             # CSS/JS assets (if any)
├── templates/          # HTML templates
├── data/               # Dataset files
├── report.pdf          # Full technical report
└── README.md           # Project documentation
```

---

## 👩‍🏫 Audience

* **Educators & School Admins:** Use predictions & sentiment insights to support students.
* **Researchers & Practitioners:** Explore the workflow from raw data to deployment.

---

## 🔗 Repository

All project files, demo, and report are available here in this repository.
