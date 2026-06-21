# Order Management System

## 1. System Overview
The Order Management System (OMS) is a full-stack, predictive fulfillment platform designed specifically for the complexities of eyewear manufacturing. The architecture is designed to be lightweight, modular, and highly responsive.

### Tech Stack
* **Frontend:** Gradio (Provides a rapid, reactive dashboard for order intake, SLA monitoring, and lifecycle management).
* **Backend:** FastAPI (Handles concurrent requests and serves as the bridge between the UI, the database, and the AI model).
* **Database:** SQLite via SQLAlchemy ORM (Ensures persistent, structured tracking of order lifecycles and delay reasons).

---

## 2. AI Model Selection & Rationale

**Task:** Supervised Binary Classification (Predicting SLA Breaches before they occur).

To ensure the highest accuracy for the predictive breach engine, a comparative analysis of several machine learning algorithms was conducted against the synthetic order dataset. The models evaluated included:
* Logistic Regression
* Support Vector Machines (SVM)
* XGBoost
* Random Forest (RF)

### Why Random Forest was Selected
After rigorous testing, the Random Forest Classifier emerged as the best-performing model for this specific use case.

1. **Handling Tabular Data:** Eyewear orders consist of a mix of categorical (Lens Type, Coating) and numerical (SPH power, Lens Index) features. RF inherently handles this mixed tabular data exceptionally well without requiring complex scaling pipelines.
2. **Performance vs. Complexity:** While XGBoost offered competitive accuracy, Random Forest achieved the highest overall performance metrics with significantly less hyperparameter tuning and lower risk of overfitting (controlled via `max_depth=10`).
3. **Non-linear Relationships:** Logistic Regression underperformed because the relationships between specific custom coatings, extreme SPH powers, and SLA delays are non-linear. SVM was computationally heavier and less interpretable.
4. **Lightweight Deployment:** The trained RF model and its label encoders serialize cleanly via `joblib`, allowing the FastAPI backend to load it in milliseconds and run real-time inference without lagging the user interface.

---

## 3. External API Integrations & Rationale
To fulfill the requirement for proactive alerts, the system integrates two external communication APIs triggered by the AI risk predictions.

* **Twilio API (WhatsApp Integration):** Floor technicians and store staff rarely sit at desks monitoring emails. Twilio was chosen to deliver instant, push-style WhatsApp notifications directly to the team's mobile devices the second a high-risk order is placed, enabling immediate operational intervention.
* **Resend API (Email Integration):** Resend provides highly reliable, low-latency email delivery. This was chosen for managerial oversight. While WhatsApp is for immediate action, the Resend email provides a formal, formatted HTML log of the predicted breach for shift managers to review and document.
