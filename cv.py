import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import xgboost as xgb
import shap

# ── 1. Load & Prepare Data ──────────────────────────────────────────────────
df = pd.read_csv('encoded_data.csv')
df = df.fillna(0)

target_col = 'Yield_Strength'
X = df.drop(columns=[target_col])
y = df[target_col]

print("=" * 60)
print("Dataset Shape:", df.shape)
print("\nTarget Distribution (Yield_Strength):")
print(y.describe())
print("=" * 60)

# ── 2. Helper: CV6 R² ───────────────────────────────────────────────────────
def cv6_r2(estimator, X_sub, y):
    scores = cross_val_score(estimator, X_sub, y, cv=6, scoring='r2')
    return round(scores.mean(), 4)

# ── 3a. Random Forest (RF) ──────────────────────────────────────────────────
print("\n[RF] Fitting...")
rf_model = RandomForestRegressor()
rf_model.fit(X, y)
rf_imp = pd.Series(rf_model.feature_importances_, index=X.columns)

rf_top6     = rf_imp.nlargest(6).index.tolist()
rf_removed  = rf_imp.idxmax()

X_rf_red    = X.drop(columns=[rf_removed])
rf_model2   = RandomForestRegressor()
rf_model2.fit(X_rf_red, y)
rf_imp2     = pd.Series(rf_model2.feature_importances_, index=X_rf_red.columns)
rf_top5     = rf_imp2.nlargest(5).index.tolist()

rf_cv6_full = cv6_r2(RandomForestRegressor(), X[rf_top6], y)
rf_cv6_red  = cv6_r2(RandomForestRegressor(), X_rf_red[rf_top5], y)
print(f"  Top6: {rf_top6}")
print(f"  Removed: {rf_removed}  |  Top5: {rf_top5}")
print(f"  CV6 Full={rf_cv6_full}  |  CV6 Reduced={rf_cv6_red}")

# ── 3b. RF-SHAP ─────────────────────────────────────────────────────────────
print("\n[RF-SHAP] Fitting...")
rf_shap_model = RandomForestRegressor()
rf_shap_model.fit(X, y)
explainer_rf      = shap.TreeExplainer(rf_shap_model)
shap_rf_values    = explainer_rf.shap_values(X)
rfshap_imp        = pd.Series(np.abs(shap_rf_values).mean(axis=0), index=X.columns)

rfshap_top6       = rfshap_imp.nlargest(6).index.tolist()
rfshap_removed    = rfshap_imp.idxmax()

X_rfshap_red      = X.drop(columns=[rfshap_removed])
rf_shap_model2    = RandomForestRegressor()
rf_shap_model2.fit(X_rfshap_red, y)
explainer_rf2     = shap.TreeExplainer(rf_shap_model2)
shap_rf_values2   = explainer_rf2.shap_values(X_rfshap_red)
rfshap_imp2       = pd.Series(np.abs(shap_rf_values2).mean(axis=0), index=X_rfshap_red.columns)
rfshap_top5       = rfshap_imp2.nlargest(5).index.tolist()

rfshap_cv6_full   = cv6_r2(RandomForestRegressor(), X[rfshap_top6], y)
rfshap_cv6_red    = cv6_r2(RandomForestRegressor(), X_rfshap_red[rfshap_top5], y)
print(f"  Top6: {rfshap_top6}")
print(f"  Removed: {rfshap_removed}  |  Top5: {rfshap_top5}")
print(f"  CV6 Full={rfshap_cv6_full}  |  CV6 Reduced={rfshap_cv6_red}")

# ── 3c. XGBoost (XGB) ───────────────────────────────────────────────────────
print("\n[XGB] Fitting...")
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)

xgb_top6     = xgb_imp.nlargest(6).index.tolist()
xgb_removed  = xgb_imp.idxmax()

X_xgb_red    = X.drop(columns=[xgb_removed])
xgb_model2   = xgb.XGBRegressor()
xgb_model2.fit(X_xgb_red, y)
xgb_imp2     = pd.Series(xgb_model2.feature_importances_, index=X_xgb_red.columns)
xgb_top5     = xgb_imp2.nlargest(5).index.tolist()

xgb_cv6_full = cv6_r2(xgb.XGBRegressor(), X[xgb_top6], y)
xgb_cv6_red  = cv6_r2(xgb.XGBRegressor(), X_xgb_red[xgb_top5], y)
print(f"  Top6: {xgb_top6}")
print(f"  Removed: {xgb_removed}  |  Top5: {xgb_top5}")
print(f"  CV6 Full={xgb_cv6_full}  |  CV6 Reduced={xgb_cv6_red}")

# ── 3d. XGB-SHAP ────────────────────────────────────────────────────────────
print("\n[XGB-SHAP] Fitting...")
xgb_shap_model  = xgb.XGBRegressor()
xgb_shap_model.fit(X, y)
explainer_xgb   = shap.TreeExplainer(xgb_shap_model)
shap_xgb_values = explainer_xgb.shap_values(X)
xgbshap_imp     = pd.Series(np.abs(shap_xgb_values).mean(axis=0), index=X.columns)

xgbshap_top6    = xgbshap_imp.nlargest(6).index.tolist()
xgbshap_removed = xgbshap_imp.idxmax()

X_xgbshap_red     = X.drop(columns=[xgbshap_removed])
xgb_shap_model2   = xgb.XGBRegressor()
xgb_shap_model2.fit(X_xgbshap_red, y)
explainer_xgb2    = shap.TreeExplainer(xgb_shap_model2)
shap_xgb_values2  = explainer_xgb2.shap_values(X_xgbshap_red)
xgbshap_imp2      = pd.Series(np.abs(shap_xgb_values2).mean(axis=0), index=X_xgbshap_red.columns)
xgbshap_top5      = xgbshap_imp2.nlargest(5).index.tolist()

xgbshap_cv6_full  = cv6_r2(xgb.XGBRegressor(), X[xgbshap_top6], y)
xgbshap_cv6_red   = cv6_r2(xgb.XGBRegressor(), X_xgbshap_red[xgbshap_top5], y)
print(f"  Top6: {xgbshap_top6}")
print(f"  Removed: {xgbshap_removed}  |  Top5: {xgbshap_top5}")
print(f"  CV6 Full={xgbshap_cv6_full}  |  CV6 Reduced={xgbshap_cv6_red}")

# ── 3e. Feature Agglomeration (FA) ──────────────────────────────────────────
print("\n[FA] Fitting...")
fa = FeatureAgglomeration(n_clusters=6)
fa.fit(X)
fa_var     = X.var(axis=0)
fa_scores  = pd.Series(fa_var.values, index=X.columns)
fa_top6    = fa_scores.nlargest(6).index.tolist()
fa_removed = fa_scores.idxmax()

X_fa_red   = X.drop(columns=[fa_removed])
fa2        = FeatureAgglomeration(n_clusters=6)
fa2.fit(X_fa_red)
fa_var2    = X_fa_red.var(axis=0)
fa_scores2 = pd.Series(fa_var2.values, index=X_fa_red.columns)
fa_top5    = fa_scores2.nlargest(5).index.tolist()

fa_cv6_full = cv6_r2(RandomForestRegressor(), X[fa_top6], y)
fa_cv6_red  = cv6_r2(RandomForestRegressor(), X_fa_red[fa_top5], y)
print(f"  Top6: {fa_top6}")
print(f"  Removed: {fa_removed}  |  Top5: {fa_top5}")
print(f"  CV6 Full={fa_cv6_full}  |  CV6 Reduced={fa_cv6_red}")

# ── 3f. Highly Variable Gene Selection (HVGS) ───────────────────────────────
print("\n[HVGS] Fitting...")
hvgs_scores  = X.var(axis=0)
hvgs_top6    = hvgs_scores.nlargest(6).index.tolist()
hvgs_removed = hvgs_scores.idxmax()

X_hvgs_red   = X.drop(columns=[hvgs_removed])
hvgs_scores2 = X_hvgs_red.var(axis=0)
hvgs_top5    = hvgs_scores2.nlargest(5).index.tolist()

hvgs_cv6_full = cv6_r2(RandomForestRegressor(), X[hvgs_top6], y)
hvgs_cv6_red  = cv6_r2(RandomForestRegressor(), X_hvgs_red[hvgs_top5], y)
print(f"  Top6: {hvgs_top6}")
print(f"  Removed: {hvgs_removed}  |  Top5: {hvgs_top5}")
print(f"  CV6 Full={hvgs_cv6_full}  |  CV6 Reduced={hvgs_cv6_red}")

# ── 3g. Spearman ────────────────────────────────────────────────────────────
print("\n[Spearman] Fitting...")
sp_scores  = {col: abs(spearmanr(X[col], y).correlation) for col in X.columns}
sp_series  = pd.Series(sp_scores)
sp_top6    = sp_series.nlargest(6).index.tolist()
sp_removed = sp_series.idxmax()

X_sp_red   = X.drop(columns=[sp_removed])
sp_scores2 = {col: abs(spearmanr(X_sp_red[col], y).correlation) for col in X_sp_red.columns}
sp_series2 = pd.Series(sp_scores2)
sp_top5    = sp_series2.nlargest(5).index.tolist()

sp_cv6_full = cv6_r2(RandomForestRegressor(), X[sp_top6], y)
sp_cv6_red  = cv6_r2(RandomForestRegressor(), X_sp_red[sp_top5], y)
print(f"  Top6: {sp_top6}")
print(f"  Removed: {sp_removed}  |  Top5: {sp_top5}")
print(f"  CV6 Full={sp_cv6_full}  |  CV6 Reduced={sp_cv6_red}")

# ── 4. Summary Table ────────────────────────────────────────────────────────
rows = [
    ('RF',       rf_cv6_full,      rf_cv6_red,      rf_top6,       rf_top5),
    ('RF-SHAP',  rfshap_cv6_full,  rfshap_cv6_red,  rfshap_top6,   rfshap_top5),
    ('XGB',      xgb_cv6_full,     xgb_cv6_red,     xgb_top6,      xgb_top5),
    ('XGB-SHAP', xgbshap_cv6_full, xgbshap_cv6_red, xgbshap_top6,  xgbshap_top5),
    ('FA',       fa_cv6_full,      fa_cv6_red,      fa_top6,        fa_top5),
    ('HVGS',     hvgs_cv6_full,    hvgs_cv6_red,    hvgs_top6,      hvgs_top5),
    ('Spearman', sp_cv6_full,      sp_cv6_red,      sp_top6,        sp_top5),
]

summary = pd.DataFrame(rows, columns=[
    'Method', 'CV6_Full_R2', 'CV6_Reduced_R2', 'Top6_Features', 'Top5_Features'
])

summary['Top6_Features'] = summary['Top6_Features'].apply(lambda x: '; '.join(x))
summary['Top5_Features'] = summary['Top5_Features'].apply(lambda x: '; '.join(x))

for col in ['CV6_Full_R2', 'CV6_Reduced_R2']:
    summary[col] = summary[col].apply(lambda v: float(f'{v:.4g}'))

# ── 5. Save result.csv (exactly 4 columns) ──────────────────────────────────
result = summary[['Method', 'CV6_Full_R2', 'Top6_Features', 'Top5_Features']].copy()
result.rename(columns={'CV6_Full_R2': 'CV6_Accuracy'}, inplace=True)

print("\n" + "=" * 60)
print("SUMMARY TABLE (4 columns saved to result.csv)")
print("=" * 60)
print(result.to_string(index=False))
result.to_csv('result.csv', index=False)
print("\nSaved --> result.csv")

# ── 6. Extended table with reduced CV6 (console only) ───────────────────────
print("\n" + "=" * 60)
print("EXTENDED TABLE (incl. CV6 Reduced R2)")
print("=" * 60)
print(summary.to_string(index=False))
