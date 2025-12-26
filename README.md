
# 📊 StatFlow AI (by Team NumerIQ)

**Advanced Automated Survey Estimation & Reporting Platform**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Flask](https://img.shields.io/badge/Framework-Flask-green) ![License](https://img.shields.io/badge/License-MIT-orange) ![Status](https://img.shields.io/badge/Status-POC_Ready-success)

StatFlow AI is a production-ready web application designed to automate the lifecycle of official survey statistics. It streamlines the complex process of ingestion, cleaning, weighted estimation, and report generation into a unified, secure, and user-friendly interface.

## 🚀 Key Features (Round 2 Updates)

### 📥 **Universal Ingestion Engine**
* **Multi-Format Support:** Ingests **CSV**, **Excel** (.xlsx), **SPSS** (.sav), and **SAS** (.sas7bdat) files via `pyreadstat`.
* **Smart Batching:** Drag-and-drop up to **5 files** simultaneously with folder-rejection logic.
* **Schema Mapping:** Interactive UI to map inconsistent headers to standard official schemas.

### 🧹 **Deep Data Integrity & Cleaning**
* **Granular Duplicate Detection:** Identifies specific row indices and exact content matches.
* **Automated Cleaning:** Missing value imputation (Mean/Median/Drop) and Outlier removal (IQR Method).
* **Rule-Based Validation:** Enforces logical constraints (e.g., Age > 0, Income >= 0).

### 📈 **Advanced Analytics & Estimation**
* **In-Browser Pivot Tables:** dynamic grouping, aggregation (Sum, Mean, Count), and summarization.
* **Rigorous Estimation:** Calculates **Weighted Means**, **Standard Errors (SE)**, **Margins of Error (MOE)**, and **95% Confidence Intervals**.
* **Interactive Visuals:** Real-time distribution charts and outlier boxplots using **Chart.js**.

### 📄 **Official Reporting**
* **AI Executive Summary:** Auto-generates natural language insights from statistical tables.
* **Government-Style Reports:** One-click download of standardized PDF/HTML reports.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS, Chart.js, Lucide Icons |
| **Backend** | Python (Flask), Flask-Session, Gunicorn |
| **Data Core** | Pandas, NumPy, Pyreadstat (SAS/SPSS), OpenPyXL |
| **Security** | Werkzeug Security, Role-Based Access Control (RBAC) |
| **Deployment** | Vercel / Render (Serverless compatible) |

---

## ⚡ Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/statflow-ai.git](https://github.com/YOUR_USERNAME/statflow-ai.git)
cd statflow-ai

```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
python app.py

```

Access the portal at: `http://127.0.0.1:5000`

* **User:** `admin`
* **Pass:** `admin123`

---

## 🌐 Deployment

### Deploy to Render (Recommended)

1. Fork this repo.
2. Create a **New Web Service** on [Render](https://render.com).
3. Connect your repo.
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `gunicorn app:app`

### Deploy to Vercel

1. Import project to [Vercel](https://vercel.com).
2. Framework Preset: **Other**.
3. Deploy (The `vercel.json` is pre-configured for Python).

---

## 📊 Methodology

The estimation engine uses the variance formula for weighted survey data:

$$ Var(\bar{y}_w) = \frac{1}{\sum w_i} \left[ \frac{n}{n-1} \sum w_i (y_i - \bar{y}_w)^2 \right] $$

Where:

*  = Survey Design Weight
*  = Variable of Interest
*  = Sample Size

---

## 👥 Team NumerIQ

* **Lead Developer:** [Your Name]
* **Team Members:** [Teammate Names]

*Built for Statathon 2025*

```