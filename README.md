# StatFlow AI - Automated Survey Estimation & Reporting Platform

**StatFlow AI** is a production-ready web application designed to automate the lifecycle of official survey statistics. It streamlines ingestion, schema mapping, weighted estimation, and report writing into a unified secure platform.

## 🚀 Key Features
* **Batch Ingestion:** Upload multiple CSV/Excel datasets simultaneously.
* **Schema Normalization:** Interactive UI to map messy headers to standard schemas.
* **Advanced Cleaning:** Automated imputation (Mean/Median), outlier removal (IQR), and **rule-based validation** (logical range checks).
* **Rigorous Estimation:** Calculates Weighted Means, Standard Errors (SE), Margins of Error (MOE), and 95% CI.
* **AI Executive Summary:** Heuristic engine that auto-generates natural language insights.
* **Official Reporting:** Generates MoSPI-style HTML/PDF reports.

## 🛠️ Tech Stack
* **Backend:** Python 3.10+, Flask, Pandas, NumPy, SciPy
* **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JS
* **Security:** Flask-Login, Werkzeug Security
* **PDF Engine:** xhtml2pdf

## ⚡ Quick Start
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Application:**
    Double-click `run.bat` or run:
    ```bash
    python app.py
    ```
3.  **Access Portal:**
    Open `http://127.0.0.1:5000`
    * **User:** `admin`
    * **Pass:** `admin123`

## 📊 Methodology
The estimation engine uses the following variance formula for weighted survey data:
$$ Var(\bar{y}_w) = \frac{1}{\sum w_i} \left[ \frac{n}{n-1} \sum w_i (y_i - \bar{y}_w)^2 \right] $$