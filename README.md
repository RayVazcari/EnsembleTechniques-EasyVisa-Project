<h1><center><font size=10>Data Science and Business Analytics</center></font><p
<center>Project 5 - Ensemble Techniques - Visa Approval Prediction Model for EasyVisa</center></h1><p

<p align="left"> 
  <a href="https://github.com/RayVazcari?tab=followers">
    <img alt="followers" title="Follow me on Github" src="https://custom-icon-badges.demolab.com/github/followers/RayVazcari?color=236ad3&labelColor=1155ba&style=for-the-badge&logo=person-add&label=Follow me on Github &logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/rayvazcari/">
    <img alt="Linkedin Profile" title="Linkedin Profile" src="https://custom-icon-badges.demolab.com/badge/-Linkedin%20Profile-blue?style=for-the-badge&logoColor=white&logo=linkedin"/></a>
</p>

---

**`| Supervised Learning | Data Visualization | Python | Data Cleaning | Univariate Analysis | Multivariate Analysis | Data Preprocessing | Exploratory Data Analysis (EDA) | Customer Profiling | Bagging Classifiers (Bagging, Random Forest) | Boosting Classifiers (AdaBoost, Gradient Boosting, XGBoost) | Stacking Classifier | Hyperparameter Tuning with GridSearchCV |`**

---

### 🧰 Languages Libraries and Tools I Used on This Project
<a href="https://jupyter.org/" target="_blank"><img align="left" alt="Jupyter" title="Jupyter" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/jupyter/jupyter-original-wordmark.svg" /></a>
<a href="https://matplotlib.org/" target="_blank"><img align="left" alt="Matplotlib" title="Matplotlib" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" /></a>
<a href="https://numpy.org/" target="_blank"><img align="left" alt="Numpy" title="Numpy" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" /></a>
<a href="https://pandas.pydata.org/" target="_blank"><img align="left" alt="Pandas" title="Pandas" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" /></a>
<a href="https://plotly.com/" target="_blank"><img align="left" alt="Plotly" title="Plotly" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/plotly/plotly-original.svg" /></a>
<a href="https://www.python.org/" target="_blank"><img align="left" alt="Python" title="Python" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" /></a>
<a href="https://code.visualstudio.com/" target="_blank"><img align="left" alt="VScode" title="VScode" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/vscode/vscode-original.svg" /></a>
<a href="https://seaborn.pydata.org/" target="_blank"><img align="left" alt="Seaborn" title="Seaborn" width="30px" style="padding-right:10px;" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" /></a>

<br />

---

### **Project Overview**

In this project, I developed a predictive machine learning model to streamline the visa certification process for EasyVisa, a consultancy supporting the Office of Foreign Labor Certification (OFLC). With a rising volume of visa applications, OFLC faces challenges in reviewing each case manually, making it difficult to efficiently identify and certify qualified candidates. My objective was to create a data-driven solution that predicts visa approval likelihood based on applicant and employer characteristics, helping OFLC prioritize cases and reduce manual review time.

I began with Exploratory Data Analysis (EDA) to uncover significant patterns within the data, examining features like education level, work experience, prevailing wage, and job type. This step helped me understand correlations and trends, providing a foundation for model development. After thorough data preprocessing, including handling missing values, encoding categorical variables, and scaling numerical features, I prepared the dataset for robust machine learning modeling.

For modeling, I applied ensemble techniques such as Bagging (Random Forest) and Boosting (AdaBoost, Gradient Boosting, and XGBoost), which are known for improving predictive accuracy by combining multiple models. I also explored a Stacking Classifier to leverage the strengths of multiple algorithms and maximize predictive performance. By utilizing GridSearchCV for hyperparameter tuning, I optimized each model, ensuring high precision in predicting visa outcomes. This process not only helped in identifying high-accuracy models but also demonstrated which factors were most influential in predicting visa approval.

### **Summary of Findings**

- `Education and Job Experience`: Higher education levels and relevant work experience were associated with increased approval rates, underscoring the value placed on skilled candidates.
- `Wage Levels`: Higher prevailing wages correlated with approval likelihood, reflecting OFLC’s commitment to wage standards and workforce protection.
- `Regional Influence`: Certain regions demonstrated higher approval rates, suggesting that regional factors could impact visa success.
- `Job Type Preference`: Full-time positions were more favorably reviewed, aligning with OFLC’s focus on job stability.

### **Impact**

The insights from this project allowed EasyVisa to provide OFLC with strategic recommendations, such as instituting minimum education and wage requirements and tailoring evaluations by region and job type. This machine learning solution not only enhances the visa certification process but also aligns with workforce standards, helping OFLC select candidates who meet U.S. labor market needs.

### **Project Outcome**

Through this project, I gained substantial experience in ensemble learning and hyperparameter tuning and demonstrated the potential of machine learning to support critical decision-making in government processes. This project highlights my ability to extract actionable business insights from data, leveraging advanced modeling techniques to address complex, high-stakes challenges.

<br />

---

<center><img src="https://easyvisa-sa.com/wp-content/uploads/2022/07/lgo.png"></center>

---

---
### Business Context

Business communities in the United States are facing high demand for human resources, but one of the constant challenges is identifying and attracting the right talent, which is perhaps the most important element in remaining competitive. Companies in the United States look for hard-working, talented, and qualified individuals both locally as well as abroad.

The Immigration and Nationality Act (INA) of the US permits foreign workers to come to the United States to work on either a temporary or permanent basis. The act also protects US workers against adverse impacts on their wages or working conditions by ensuring US employers' compliance with statutory requirements when they hire foreign workers to fill workforce shortages. The immigration programs are administered by the Office of Foreign Labor Certification (OFLC).

OFLC processes job certification applications for employers seeking to bring foreign workers into the United States and grants certifications in those cases where employers can demonstrate that there are not sufficient US workers available to perform the work at wages that meet or exceed the wage paid for the occupation in the area of intended employment.

### Objective

In FY 2016, the OFLC processed 775,979 employer applications for 1,699,957 positions for temporary and permanent labor certifications. This was a nine percent increase in the overall number of processed applications from the previous year. The process of reviewing every case is becoming a tedious task as the number of applicants is increasing every year.

The increasing number of applicants every year calls for a Machine Learning based solution that can help in shortlisting the candidates having higher chances of VISA approval. OFLC has hired your firm EasyVisa for data-driven solutions. You as a data scientist have to analyze the data provided and, with the help of a classification model:

- Facilitate the process of visa approvals.
- Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status.

### Data Description

The data contains the different attributes of the employee and the employer. The detailed data dictionary is given below.

**Data Dictionary**

- `case_id`: ID of each visa application
- `continent`: Information of continent the employee
- `education_of_employee`: Information of education of the employee
- `has_job_experience`: Does the employee has any job experience? Y= Yes; N = No
- `requires_job_training`: Does the employee require any job training? Y = Yes; N = No
- `no_of_employees`: Number of employees in the employer's company
- `yr_of_estab`: Year in which the employer's company was established
- `region_of_employment`: Information of foreign worker's intended region of employment in the US.
- `prevailing_wage`: Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment.
- `unit_of_wage`: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
- `full_time_position`: Is the position of work full-time? Y = Full Time Position; N = Part Time Position
- `case_status`: Flag indicating if the Visa was certified or denied
