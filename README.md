# Marketplace Seller Risk & Loss Prevention

## Research Question

How can marketplace payment platforms augment traditional rule-based loss prevention systems with a unified machine learning framework to accurately identify high-risk sellers, estimate potential financial losses, and forecast portfolio-level risk before seller payouts are released?

## Data Source

The primary dataset for this project is the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), containing over 100,000 e-commerce orders with information about sellers, customers, payments, products, delivery performance, and customer reviews.

The raw transactional data was transformed into a **seller-level dataset** (one record per seller) by engineering behavioral features including: seller tenure, order volume, average order value, payment behavior, delivery delays, customer review scores, cancellation rates, monthly sales trends, and seller activity patterns.

## Techniques

This project combines unsupervised learning, supervised learning, and time-series forecasting into a comprehensive risk assessment framework:

- **Seller Segmentation** — PCA + K-Means (planned, Module 24) to identify natural seller behavior groups.
- **Risk Prediction** — Logistic Regression (official baseline, complete) and Random Forest (comparison model, complete) this module; Decision Tree and XGBoost comparison planned for Module 24. Evaluated with Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix. Class imbalance (approx. 20% high-risk sellers) is addressed via class weighting, with SMOTE comparison planned for Module 24.
- **Portfolio Loss Forecasting** — ARIMA and Prophet (planned, Module 24) to estimate portfolio-level financial losses over 30/60/90-day horizons.

## Repository Structure

```
├── README.md
├── Capstone_EDA_Module20.ipynb   # Data cleaning, EDA, feature engineering, baseline model
└── data/
    ├── olist_customers_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

## Results (Module 20.1 — Initial Report & EDA)

### Data Cleaning
Nine raw Olist tables (99,441 orders; 3,095 sellers) were cleaned and merged into an order-level master table, then aggregated into a **3,088-row seller-level feature set**. Missing values were handled contextually rather than uniformly dropped: missing product categories (610 rows) were labeled `unknown`, missing delivery dates were preserved as a meaningful signal of non-delivery/cancellation rather than imputed, and numeric product dimensions were median-imputed. No duplicate rows were found in any source table.

### Target Variable
Olist has no explicit fraud/chargeback label, so a **composite proxy risk target** was constructed from three operational signals a marketplace risk team already monitors: cancellation rate (40% weight), late-delivery rate (30%), and negative-review rate (30%). Sellers in the top quintile of this weighted score are flagged `high_risk_seller = 1`, producing a realistic ~80/20 class split (2,468 low-risk / 620 high-risk sellers).

### Key EDA Findings
- Order status is dominated by `delivered` (~97%); cancellations and unavailable orders combined are only ~1.2% of all orders, confirming risk events are genuinely rare — the reason a percentile-based proxy target was necessary rather than a naturally occurring split.
- Late delivery, cancellation rate, and negative reviews are moderately-to-strongly correlated (r ≈ 0.3–0.6), validating them as a coherent composite risk signal, while also confirming they can't be used as independent model predictors of each other.
- High-risk and low-risk sellers show almost no separation on average order value — pricing/order size alone is not a useful risk signal.
- Risk rates vary meaningfully by seller state (roughly 9%–23% across the top 10 states by seller count), suggesting geography carries real, if modest, predictive signal — likely tied to logistics/delivery distance.

### Baseline Model
A **Logistic Regression** model (`class_weight='balanced'`) is the official Module 20 baseline, with a **Random Forest** as a comparison model. The three features used to build the target (`pct_canceled`, `pct_late`, `pct_low_reviews`) and their direct components were **excluded from the feature matrix** to prevent label leakage — both models learn from 41 independent behavioral/geographic predictors only. The dataset was split 75/25 (2,316 train / 772 test) using a stratified split to preserve class balance.

| Metric | Logistic Regression | Random Forest | Better Result |
|---|---:|---:|---|
| Accuracy | 0.738 | 0.794 | Random Forest |
| High-risk precision | 0.401 | 0.489 | Random Forest |
| High-risk recall | 0.613 | 0.581 | Logistic Regression |
| High-risk F1-score | 0.485 | 0.531 | Random Forest |
| ROC-AUC | 0.760 | 0.810 | Random Forest |
| PR-AUC | 0.445 | 0.491 | Random Forest |

**Evaluation metric rationale:** accuracy is misleading under class imbalance (a model predicting "low risk" for everyone would score ~80%). Recall, precision, F1, ROC-AUC, and PR-AUC on the high-risk class were prioritized because in a loss-prevention context, a missed high-risk seller (false negative) is generally costlier than an extra manual review (false positive).

**Model selection:** Logistic Regression remains the **official baseline** for its simplicity and interpretability. Random Forest performs better on most metrics (accuracy, precision, F1, ROC-AUC, PR-AUC), suggesting non-linear feature interactions carry real signal — but Logistic Regression achieves the higher high-risk recall, catching ~5 more true high-risk sellers in the test set. Which model is "better" depends on whether the business prioritizes minimizing missed risky sellers (favor Logistic Regression / higher recall) or minimizing false alarms (favor Random Forest / higher precision).

**Leading predictors** (standardized Logistic Regression coefficients): average delivery time (strongest single predictor), order volume (negatively associated with risk — higher-volume sellers tend to be more reliable), review count, and seller tenure/activity span.

**Important caveat:** these results apply to the constructed operational-risk proxy, not validated fraud or financial-loss labels. Additional validation, threshold tuning, time-based (out-of-time) testing, and comparison against confirmed loss data would be required before any operational use.

### Next Steps (Module 24)
- Add PCA + K-Means seller segmentation to complement the supervised risk model.
- Extend model comparison to Decision Tree and XGBoost, with formal SMOTE-based resampling.
- Build the ARIMA/Prophet portfolio-level loss forecasting layer for 30/60/90-day horizons.
- Clean up notebook code and prepare final presentation materials for technical and non-technical audiences.

## Notebook

See [`Capstone_EDA_Module20_Sanchitag.ipynb`](https://github.com/sanchitag22/CapstoneProject_UCB_MarketplaceSellerRiskandLossPrevention/blob/main/Capstone_EDA_Module20_Sanchitag.ipynb) for the full analysis, code, visualizations, and section-by-section observations.
