# 🐙 Team GitHub Collaboration & Git Workflow Guide
### Project: Climate-Smart Agriculture — Bi-Seasonal Paddy Yield Forecasting
**Module:** IT41033 Nature Inspired Algorithms (NIA) — Horizon Campus

This guide provides step-by-step Git instructions for all **5 group members** so that each member's individual commits, branches, and Pull Requests reflect their designated project role on GitHub.

---

## 📋 Overview of Roles & Branch Allocations

| Member | ID | Designated Role | Branch Name | Key Files Responsible For |
|---|---|---|---|---|
| **E. Dasun Manjitha** | ITBIN-2313-0062 | **Pipeline Architecture Lead** | `main` & `feature/scaffold-dashboard` | Repo setup, `.gitignore`, `requirements.txt`, `src/data_loader.py`, `dashboard/app.py`, `README.md` |
| **W.G.C.M. Nimsara** | ITBIN-2313-0072 | **Data Quality & Preprocessing Specialist** | `feature/data-preprocessing` | `src/preprocessing.py`, `notebooks/02_preprocessing.ipynb`, `data/processed/model_ready.csv` |
| **R.T. Dinith Sasanga** | ITBIN-2313-0101 | **Categorical Engineering Specialist** | `feature/feature-engineering` | `src/features.py`, `notebooks/03_feature_engineering.ipynb`, `data/processed/model_ready_features.csv` |
| **W.A.S.I. Wijesinghe** | ITBIN-2313-0129 | **Machine Learning Modeling Engineer** | `feature/ml-modeling-evaluation` | `src/models.py`, `notebooks/04_modeling.ipynb`, `notebooks/05_evaluation.ipynb`, `tests/test_pipeline.py` |
| **R.G.D.N. Wijesuriya** | ITBIN-2313-0130 | **Descriptive Statistics & Visualization Lead** | `feature/eda-visualizations-report` | `src/eda.py`, `notebooks/01_eda.ipynb`, `reports/figures/`, `reports/final_report.md` |

---

## 🚀 STEP 0: Initial GitHub Repository Creation (Lead: Dasun)

1. Go to [GitHub.com](https://github.com) and click **New Repository**.
2. Name the repository: `SL-Climate-Smart-Agriculture` (or `paddy-yield-forecasting`).
3. Choose **Public** (or Private and add all 4 teammates as Collaborators under **Settings > Collaborators**).
4. Do **NOT** initialize with README or .gitignore (we already have them in our project folder).
5. Copy your repository URL (e.g., `https://github.com/YourUsername/SL-Climate-Smart-Agriculture.git`).

---

## 👤 MEMBER 1: E. Dasun Manjitha (Pipeline Architecture Lead)
**Goal:** Initialize the remote, push base structure, requirements, data loader, and Streamlit dashboard.

Open PowerShell in `d:\SL-Climate-Smart Agriculture` and run:

```bash
# 1. Check git status
git status

# 2. Add base scaffold files
git add .gitignore requirements.txt README.md src/__init__.py src/data_loader.py dashboard/app.py

# 3. Commit scaffold
git commit -m "feat(arch): initial repository scaffold, data ingestion loader, and Streamlit dashboard"

# 4. Set main branch and link GitHub remote
git branch -M main
git remote add origin https://github.com/YourUsername/SL-Climate-Smart-Agriculture.git

# 5. Push to GitHub
git push -u origin main
```

---

## 👥 INSTRUCTIONS FOR ALL OTHER TEAMMATES (Nimsara, Dinith, Wijesinghe, Wijesuriya)

Each teammate clones the repository onto their PC, creates their assigned feature branch, commits their files, pushes to GitHub, and creates a Pull Request.

---

## 👤 MEMBER 2: W.G.C.M. Nimsara (Data Quality & Preprocessing Specialist)
**Assigned Files:** `src/preprocessing.py`, `notebooks/02_preprocessing.ipynb`, `data/processed/model_ready.csv`

```bash
# 1. Clone repository (or pull latest)
git clone https://github.com/YourUsername/SL-Climate-Smart-Agriculture.git
cd SL-Climate-Smart-Agriculture

# 2. Create and switch to your feature branch
git checkout -b feature/data-preprocessing

# 3. Add your assigned files
git add src/preprocessing.py notebooks/02_preprocessing.ipynb data/processed/model_ready.csv

# 4. Commit with descriptive message
git commit -m "feat(preprocessing): implement IQR winsorization, season binarization, and leakage-free scaling"

# 5. Push branch to GitHub
git push -u origin feature/data-preprocessing
```

---

## 👤 MEMBER 3: R.T. Dinith Sasanga (Categorical Engineering Specialist)
**Assigned Files:** `src/features.py`, `notebooks/03_feature_engineering.ipynb`, `data/processed/model_ready_features.csv`

```bash
# 1. Update main and branch off
git checkout main
git pull origin main
git checkout -b feature/feature-engineering

# 2. Add your assigned files
git add src/features.py notebooks/03_feature_engineering.ipynb data/processed/model_ready_features.csv

# 3. Commit with descriptive message
git commit -m "feat(features): add autoregressive lags, 3-season rolling climate means, and drought anomaly flag"

# 4. Push branch to GitHub
git push -u origin feature/feature-engineering
```

---

## 👤 MEMBER 4: W.A.S.I. Wijesinghe (Machine Learning Modeling Engineer)
**Assigned Files:** `src/models.py`, `notebooks/04_modeling.ipynb`, `notebooks/05_evaluation.ipynb`, `tests/test_pipeline.py`

```bash
# 1. Update main and branch off
git checkout main
git pull origin main
git checkout -b feature/ml-modeling-evaluation

# 2. Add your assigned files
git add src/models.py notebooks/04_modeling.ipynb notebooks/05_evaluation.ipynb tests/test_pipeline.py

# 3. Commit with descriptive message
git commit -m "feat(models): implement TimeSeriesSplit CV, tune Random Forest/LightGBM/MLP, and test suite"

# 4. Push branch to GitHub
git push -u origin feature/ml-modeling-evaluation
```

---

## 👤 MEMBER 5: R.G.D.N. Wijesuriya (Descriptive Statistics & Visualization Lead)
**Assigned Files:** `src/eda.py`, `notebooks/01_eda.ipynb`, `reports/figures/`, `reports/final_report.md`, `reports/individual_contribution_template.md`

```bash
# 1. Update main and branch off
git checkout main
git pull origin main
git checkout -b feature/eda-visualizations-report

# 2. Add your assigned files
git add src/eda.py notebooks/01_eda.ipynb reports/figures/ reports/final_report.md reports/individual_contribution_template.md

# 3. Commit with descriptive message
git commit -m "feat(eda): add EDA analysis, diagnostic figures, final report draft, and contribution matrix"

# 4. Push branch to GitHub
git push -u origin feature/eda-visualizations-report
```

---

## 🔀 STEP 5: Pull Request & Merge Workflow on GitHub

1. On **GitHub.com**, go to the repository page.
2. Click **Pull requests > New pull request**.
3. Create a Pull Request for each branch (`feature/data-preprocessing`, `feature/feature-engineering`, `feature/ml-modeling-evaluation`, `feature/eda-visualizations-report`) into `main`.
4. Team members review and click **Merge pull request > Confirm merge**.
5. Once all 4 PRs are merged, all 5 members run:
   ```bash
   git checkout main
   git pull origin main
   ```
   Now everyone has the synchronized, complete codebase with full individual contribution records on GitHub! 🎉
