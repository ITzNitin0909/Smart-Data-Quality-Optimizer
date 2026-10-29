# 🔍 Smart Data Quality Optimizer

An AI-powered data quality analysis application built using Python, Streamlit, Pandas, and Scikit-learn.

## 📌 Project Overview

The Smart Data Quality Optimizer analyzes CSV datasets to identify potential data quality issues using rule-based validation and machine learning-based anomaly detection.

It helps users inspect missing values, detect duplicate records, flag unusual transactions, compare detection methods, and download analysis reports.

## ✨ Features

- **CSV Upload:** Upload and analyze CSV datasets.
- **Dataset Overview:** View total rows, columns, duplicates, and missing values.
- **Missing Value Analysis:** Identify missing values by column and generate recommendations.
- **Rule-Based Validation:** Detect duplicate rows, negative quantities, zero prices, and negative prices.
- **AI Anomaly Detection:** Apply Isolation Forest to Quantity and Price features.
- **AI vs Rule-Based Comparison:** Compare records flagged by both methods and records flagged by only one method.
- **Anomaly Pattern Analysis:** Analyze quantity and price patterns among AI-detected anomalies.
- **Downloadable Reports:** Export analysis results as CSV files.
- **Interactive Dashboard:** Explore results using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Web dashboard |
| Pandas | Data processing and analysis |
| Scikit-learn | Isolation Forest algorithm |
| CSV | Dataset input and report export |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ITzNitin0909/Smart-Data-Quality-Optimizer.git
cd Smart-Data-Quality-Optimizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install streamlit pandas scikit-learn
```

### 5. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at `http://localhost:8501`.

## 🔄 How It Works

1. Upload a CSV dataset.
2. Review dataset statistics and missing values.
3. Examine rule-based data quality checks.
4. Set the expected anomaly proportion.
5. Run Isolation Forest anomaly detection.
6. Compare AI anomalies with rule-based issues.
7. Download reports for further analysis.

## 🤖 Machine Learning Approach

This project uses **Isolation Forest**, an unsupervised machine learning algorithm, to identify unusual records based on numeric `Quantity` and `Price` features.

The contamination parameter controls the expected proportion of anomalies identified by the model. It does not measure model accuracy or confirm that a record is incorrect.

## 📊 Dataset

The application has been tested with a retail transaction dataset containing columns such as:

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

For rule-based checks and AI anomaly detection, the uploaded dataset must contain `Quantity` and `Price` columns.

## ⚠️ Important Notes

- Anomalies are potential outliers, not confirmed errors.
- Negative quantities and zero prices may be valid business transactions.
- The data quality score is provisional and depends on the selected rules.
- Results should be reviewed before changing or deleting data.

## 🔮 Future Enhancements

- Automated data cleaning with user approval
- Additional validation rules
- More anomaly detection algorithms
- Cloud deployment
- Automated testing and CI/CD integration
- Monitoring and data quality trend analysis

## 👨‍💻 Author

**Nitin Kumar**

- GitHub: [ITzNitin0909](https://github.com/ITzNitin0909)
- LinkedIn: [Nitin Kumar](https://www.linkedin.com/in/nitin-kumar-7b55882b2/)

---

*Developed as a practical project in data quality analysis and machine learning.*
