## Marketplace Seller Risk & Loss Prevention

**Sanchita Gawand**

### Executive Summary

**Project overview and goals:** 

Online marketplaces pay sellers for orders before all the outcomes of that sale are known — will it be delivered on time, will the customer be happy, will it get canceled or refunded? Traditional loss-prevention systems rely on hand-written rules ("flag any seller with more than X cancellations"), which are simple but slow to adapt and easy to miss. This project builds a machine learning framework that instead *learns* which seller behavior patterns are associated with risk, so that a marketplace payment platform can flag likely problem sellers **before** releasing their payout, rather than after the fact.

Using real transaction data from Olist, a Brazilian e-commerce marketplace, we engineered a seller-level dataset capturing each seller's order volume, delivery speed, review scores, cancellation history, and other behavioral patterns. We then built and compared four machine learning models to predict which sellers are likely to be high-risk, and used unsupervised clustering to uncover natural seller segments that a risk team could use to design different monitoring policies.

**Findings:** 

The best-performing model is a **Random Forest classifier**, correctly identifying roughly 68% of genuinely high-risk sellers in a held-out test set, with an ROC-AUC of 0.81 (a common measure of how well a model ranks risky sellers above safe ones, where 0.5 is random guessing and 1.0 is perfect). 
This is a meaningful improvement over an earlier, simpler baseline model, which caught only 61% of high-risk sellers.

The single strongest predictor of seller risk, by a wide margin, is **average delivery time** — sellers who consistently ship slower than average are far more likely to be flagged high-risk. Interestingly, how long a seller has been active on the platform turned out to matter far less than initially expected once delivery behavior was accounted for — a finding only visible after applying a more rigorous feature-importance check (see Results below).

**Results and conclusion:** 

Clustering the sellers by behavior revealed two natural groups: a smaller set of high-volume, "established" sellers, and a larger set of smaller, newer sellers. Counterintuitively, the established, high-volume group has **nearly double the risk rate** of the smaller group (33% vs. 17%) — meaning scale and tenure should not be treated as a proxy for safety. This is one of the most actionable findings of the project: a risk team's intuition that "bigger, more established sellers are safer" does not hold up in this data.

**Next steps and recommendations:** 

The current model relies on a constructed proxy for risk (built from cancellation rate, late deliveries, and negative reviews), since the dataset has no confirmed fraud or chargeback labels. Before this model informs real payout decisions, it should be validated against actual loss data if and when the business can provide it. We also recommend removing a redundant feature discovered during interpretation, testing the model's performance on more recent (out-of-time) sellers rather than a random historical sample, and piloting a tiered review process that uses the model's confidence score rather than a single yes/no flag.

### Rationale

Marketplace platforms extend a form of credit to their sellers: they collect payment from customers and forward it to sellers, often before delivery is confirmed or a return window has closed. If a seller turns out to be unreliable or fraudulent, the platform can be left absorbing the loss. Rule-based systems ("if cancellation rate > 10%, flag") are easy to understand but blunt — they can't easily weigh several weak signals together, they don't adapt as seller behavior shifts, and they tend to lag real problems rather than anticipate them.

A machine learning approach can combine many weaker signals — delivery speed, order history, review patterns, and more — into a single, tunable risk score, and can be re-evaluated and retrained as new data comes in. This project explores whether such an approach is viable using real, publicly available marketplace data.

### Research Question

How can marketplace payment platforms augment traditional rule-based loss prevention systems with a unified machine learning framework to accurately identify high-risk sellers, estimate potential financial losses, and forecast portfolio-level risk before seller payouts are released?

### Data Sources

**Dataset:** This project uses the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), a public dataset on Kaggle containing 99,441 real e-commerce orders placed between 2016 and 2018, along with information on customers, sellers, products, payments, and reviews.

**Exploratory data analysis:** 

The order data is dominated by successfully delivered orders (about 97%); cancellations and unavailable orders combined make up only about 1.2% of all orders — confirming that "risk events" are genuinely rare, which shaped how we defined a usable target variable (see Methodology). Order volume grew steadily from the platform's 2016 launch through a late-2017 peak (consistent with Black Friday seasonality), then leveled off through mid-2018.

**Cleaning and preparation:** 

Nine raw data tables were merged and cleaned, then aggregated from the order level up to **one row per seller** (3,088 sellers), since the business question is about *seller* risk, not individual order risk. Missing values were handled based on what they actually meant in context — for example, a missing delivery date usually meant the order was never delivered (a meaningful signal, not something to fill in), while a missing product category was simply labeled "unknown."

**Final dataset:** 

The seller-level dataset contains 33 engineered behavioral features — including tenure, order volume, average order value, payment behavior, delivery speed, review scores, cancellation rate, monthly sales trends, and activity patterns — for 3,088 sellers.

**The target variable:** 

Since this dataset has no confirmed fraud or chargeback label, we built a **proxy risk score** from three signals a real marketplace risk team already monitors: cancellation rate, late-delivery rate, and negative-review rate. Sellers in the riskiest 20% by this composite score are labeled "high-risk." This produces a realistic, imbalanced split (about 80% low-risk, 20% high-risk sellers) similar to real-world fraud/loss rates.

### Methodology

**Seller Segmentation:** 

We used Principal Component Analysis (PCA) to reduce the seller behavioral features to their core underlying patterns, then applied K-Means clustering to identify natural seller groupings. The optimal number of clusters (2) was chosen using the silhouette score, a standard statistical measure of how well-separated clusters are.

**Risk Prediction:** 

Four classification models were trained and compared: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Each was tuned using **GridSearchCV with 5-fold cross-validation** — rather than manually guessing hyperparameters, this systematically tests many combinations and picks the ones that generalize best across five different train/validation splits, reducing the risk of a model that looks good by chance on just one split. Class imbalance (~80/20 low-risk to high-risk) was addressed using class weighting, which tells the model to pay proportionally more attention to the rarer high-risk class during training.

**Feature leakage guard:** 

The three signals used to build the proxy target (cancellation rate, late-delivery rate, negative-review rate) were deliberately excluded from the model's input features. Without this precaution, a model could simply learn to reconstruct its own target definition rather than learning genuinely predictive patterns — a subtle but critical mistake we caught and corrected during development.

**Evaluation metric rationale:** 

Because the target is imbalanced (~80% of sellers are low-risk), plain accuracy is a misleading metric — a model that flags nobody as high-risk would still be "80% accurate" while being completely useless. We instead prioritized **recall, precision, F1-score, and ROC-AUC on the high-risk class**, since in a loss-prevention context, missing a genuinely risky seller (a false negative) is typically more costly than flagging a safe seller for extra review (a false positive).

**Model selection:** 

Rather than picking the model with the single best test-set score, we selected the model with the best performance *after accounting for cross-validation stability* — a model whose score barely edges out another but swings wildly across different data splits is a less trustworthy choice for production than one with slightly lower but much more consistent performance.

### Model Evaluation and Results

Four models were compared using GridSearchCV-tuned hyperparameters and evaluated on a held-out test set of 772 sellers (25% of the data, stratified to preserve the risk-class balance):

| Model | Accuracy | High-Risk Precision | High-Risk Recall | High-Risk F1 | ROC-AUC | 5-Fold CV F1 (mean ± std) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.738 | 0.401 | 0.613 | 0.485 | 0.760 | 0.505 ± 0.010 |
| Decision Tree | 0.750 | 0.416 | 0.606 | 0.493 | 0.759 | 0.495 ± 0.033 |
| **Random Forest (recommended)** | 0.764 | 0.444 | 0.684 | 0.538 | 0.806 | 0.541 ± 0.018 |
| Gradient Boosting | 0.767 | 0.447 | 0.684 | 0.541 | 0.809 | 0.397 ± 0.094 |

**Why Random Forest, not Gradient Boosting?** 

Gradient Boosting technically edges out Random Forest on this one test split (F1 0.541 vs. 0.538), but its cross-validation results swing far more (standard deviation of 0.094, nearly five times higher than Random Forest's 0.018). That instability is a warning sign that Gradient Boosting's edge on this particular split may not hold up reliably on new data. Random Forest offers essentially the same performance with much greater consistency, making it the more defensible choice for a real deployment.

**Feature importance:** 

The model's built-in importance ranking initially suggested seller tenure was a top-3 predictor — but a more rigorous check (permutation importance, which measures the actual performance drop when a feature is scrambled) showed tenure barely mattered at all once delivery behavior was accounted for. **Average delivery time is, by a wide margin, the single strongest predictor of seller risk** in this dataset; order-value variability, total revenue, and review count offer modest secondary signal.

**Seller segmentation:** 

Clustering revealed two natural groups — a smaller set of high-volume, established sellers (681, ≈22%) and a larger set of smaller, newer sellers (2,407, ≈78%). The established group has a substantially *higher* high-risk rate (≈33%) than the smaller group (≈17%), directly challenging the intuitive assumption that scale and tenure imply safety.

### Outline of Project

- [Main analysis notebook — data cleaning, EDA, feature engineering, segmentation, and model comparison](./Capstone_Main_Modeling.ipynb)
- [Model evaluation notebook — feature importance, example predictions, business interpretation, and next steps](./Capstone_Model_Evaluation.ipynb)
- [Dataset (Olist Brazilian E-Commerce Dataset on Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**Note on run order:** 

The evaluation notebook loads a small file (`artifacts/modeling_artifacts.pkl`) that the main modeling notebook saves at the end of its model comparison section. **Run `Capstone_Main_Modeling.ipynb` first**, then `Capstone_Model_Evaluation.ipynb` — running the evaluation notebook on its own, without the main notebook having run first, will raise a file-not-found error.

### Limitations and Future Work

- The target variable is a **constructed proxy** for risk, not confirmed financial-loss or fraud data — all results describe how well the model predicts this proxy, not real-world losses directly.
- The final feature set unintentionally includes both a raw and an outlier-capped version of delivery time, which are highly correlated (r ≈ 0.90); a cleaner rebuild should keep only one.
- The current train/test split is a random historical sample rather than a true out-of-time test, which is the more realistic way to validate a model intended for future deployment.
- Planned future work includes replacing the proxy target with real loss data if available, adding XGBoost and formal SMOTE-based resampling to the model comparison, and building an ARIMA/Prophet-based forecasting layer to estimate portfolio-level losses over 30/60/90-day horizons — extending this project from "is this seller risky" to "how much loss should we expect across the whole portfolio."

### Contact and Further Information

Sanchita Gawand
