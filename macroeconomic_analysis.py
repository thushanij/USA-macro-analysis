import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "C:/Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {"blue":"#0f3460","red":"#e94560","green":"#06d6a0","gold":"#f5a623",
     "teal":"#00b4d8","purple":"#7b2d8b","text":"#2d3436","light":"#f8f9fa"}

plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"white","axes.facecolor":"#fafafa",
    "axes.grid":True,"grid.alpha":0.35,"grid.linestyle":"--",
})

EVENTS = {"1973-10":"Oil\nShock","1979-10":"Iran\nCrisis","2000-03":"Dot-com\nBust",
          "2008-09":"GFC","2020-03":"COVID","2022-03":"Fed\nHikes"}

def annotate_events(ax, idx, ymin, ymax, fs=7):
    for ds, lbl in EVENTS.items():
        dt = pd.Timestamp(ds)
        if idx.min() <= dt <= idx.max():
            ax.axvline(dt, color=C["gold"], lw=0.9, ls=":", alpha=0.8)
            ax.text(dt, ymax*0.97, lbl, fontsize=fs, color=C["gold"],
                    ha="center", va="top", backgroundcolor="white", alpha=0.9)


# =========================================================
# LIMITATIONS
# =========================================================
print("""
  ⚠ LIMITATIONS
• GDP monthly values are interpolated.
• Recession labels follow NBER dates.
• Models are for educational purposes only.
""")
print("="*65)
print("  1. LOAD & PROCESS DATA")
print("="*65)


# =========================================================
# 1. LOAD & PROCESS
# =========================================================

raw_gdp = pd.read_csv("GDP.csv",        parse_dates=["observation_date"], index_col="observation_date")
raw_cpi = pd.read_csv("CPIAUCSL.csv",   parse_dates=["observation_date"], index_col="observation_date")
raw_unr = pd.read_csv("UNRATE.csv",     parse_dates=["observation_date"], index_col="observation_date")
raw_fed = pd.read_csv("FEDFUNDS.csv",   parse_dates=["observation_date"], index_col="observation_date")


raw_gdp.columns=["GDP"]; raw_cpi.columns=["CPI"]; raw_unr.columns=["Unemployment"]
raw_fed.columns=["InterestRate"];
# CPI → YoY inflation rate (economically meaningful, no leakage)
raw_cpi["Inflation"] = raw_cpi["CPI"].pct_change(12) * 100
inflation_m    = raw_cpi[["Inflation"]].resample("ME").mean()

# GDP quarterly → monthly interpolation (synthetic; see LIMITATIONS)
gdp_m          = raw_gdp.resample("ME").interpolate(method="linear")

unemployment_m = raw_unr.resample("ME").mean()
interest_m     = raw_fed.resample("ME").mean()


df = gdp_m.join([inflation_m, unemployment_m, interest_m], how="inner").dropna()
print(f"  Long-history df: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")


# =========================================================
# 2. NBER RECESSION LABELS
# =========================================================
print("\n  2. NBER RECESSION LABELS")

NBER_RECESSIONS = [
    ("1957-08-01","1958-04-01"),   # Eisenhower recession
    ("1960-04-01","1961-02-01"),   # Rolling adjustment
    ("1969-12-01","1970-11-01"),   # Nixon recession
    ("1973-11-01","1975-03-01"),   # Oil shock
    ("1980-01-01","1980-07-01"),   # Volcker shock pt1
    ("1981-07-01","1982-11-01"),   # Volcker shock pt2
    ("1990-07-01","1991-03-01"),   # Gulf War recession
    ("2001-03-01","2001-11-01"),   # Dot-com bust
    ("2007-12-01","2009-06-01"),   # Global Financial Crisis
    ("2020-02-01","2020-04-01"),   # COVID-19
]

nber = pd.Series(0, index=df.index, name="Recession", dtype=int)
for s, e in NBER_RECESSIONS:
    mask = (nber.index >= pd.Timestamp(s)) & (nber.index <= pd.Timestamp(e))
    nber[mask] = 1

df["Recession"] = nber
rec_months = df["Recession"].sum()
print(f"  NBER recession months : {rec_months} ({df['Recession'].mean():.1%})")
print(f"  Recessions covered    : {len(NBER_RECESSIONS)} episodes")

# =========================================================
# 3. FEATURE ENGINEERING  (all lags — no leakage)
# =========================================================
print("\n  3. FEATURE ENGINEERING  [Created lagged variables and rolling statistics.]")

# NOTE: We deliberately use only lagged features so that at any point t the
#       model sees only information available before t. This eliminates the
#       look-ahead leakage that would arise from contemporaneous values.
for col in ["GDP","Inflation","Unemployment","InterestRate"]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag3"] = df[col].shift(3)
    df[f"{col}_lag6"] = df[col].shift(6)

df["GDP_growth"]   = df["GDP"].pct_change()
df["Infl_change"]  = df["Inflation"].diff()
df["Rate_change"]  = df["InterestRate"].diff()
df["Unemp_change"] = df["Unemployment"].diff()
df["GDP_MA12"]     = df["GDP"].rolling(12).mean()
df["Infl_MA6"]     = df["Inflation"].rolling(6).mean()
df["Unemp_MA6"]    = df["Unemployment"].rolling(6).mean()
df["Rate_MA6"]     = df["InterestRate"].rolling(6).mean()
df["Infl_vol12"]   = df["Inflation"].rolling(12).std()
df["Rate_vol12"]   = df["InterestRate"].rolling(12).std()
df["Rate_12M_Change"] = df["InterestRate"].diff(12)  # proxy for curve steepness

df.dropna(inplace=True)
print(f"  Final df: {len(df)} rows × {df.shape[1]} cols")
print(f"  Recession: {df['Recession'].sum()} months ({df['Recession'].mean():.1%})")


# =========================================================
# 4. EDA DASHBOARD
# =========================================================
print("\n  4. EDA DASHBOARD")

fig = plt.figure(figsize=(20,24))
fig.patch.set_facecolor("white")
fig.text(0.5,0.985,"USA MACROECONOMIC ANALYSIS — EDA DASHBOARD",
         ha="center",fontsize=19,fontweight="bold",color=C["blue"])
fig.text(0.5,0.972,
         "1954–2026  |  NBER Recession Dates  |  YoY Inflation  |  GDP Interpolated  |  Monthly Observations",
         ha="center",fontsize=11,color=C["text"])

gs = gridspec.GridSpec(4,3,figure=fig,top=0.965,bottom=0.04,hspace=0.55,wspace=0.38)
rec = df["Recession"]==1

# 4a GDP
ax = fig.add_subplot(gs[0,:2])
ax.plot(df.index, df["GDP"], color=C["blue"], lw=2)
ax.fill_between(df.index, df["GDP"].min(), df["GDP"].max(),
                where=rec, alpha=0.2, color=C["red"], label="NBER Recession")
annotate_events(ax, df.index, df["GDP"].min(), df["GDP"].max())
ax.set_title("GDP — Interpolated Monthly  (NBER shading)\n"
             "⚠ Monthly values are linearly interpolated from quarterly FRED data",
             fontweight="bold", color=C["blue"])
ax.set_ylabel("GDP (Billions USD)"); ax.legend(fontsize=9)

# 4b Recession pie
ax = fig.add_subplot(gs[0,2])
r = df["Recession"].sum(); total = len(df)
wedges,texts,autos = ax.pie([r,total-r],labels=["Recession","Expansion"],
    colors=[C["red"],C["green"]],autopct="%1.1f%%",startangle=140,
    wedgeprops={"edgecolor":"white","linewidth":2})
for a in autos: a.set_fontsize(11); a.set_fontweight("bold")
ax.set_title(f"NBER Recession Months\n{r} of {total} ({r/total:.1%})\n[10 official NBER episodes]",
             fontweight="bold",color=C["blue"],fontsize=9)

# 4c Inflation YoY
ax = fig.add_subplot(gs[1,0])
ax.plot(df.index, df["Inflation"], color=C["red"], lw=1.8)
ax.fill_between(df.index, 0, df["Inflation"],
                where=df["Inflation"]>4, alpha=0.2, color=C["red"], label=">4% elevated")
ax.axhline(2,color=C["gold"],ls="--",lw=1.5,label="Fed 2% target")
ax.axhline(0,color=C["text"],lw=0.8,alpha=0.4)
annotate_events(ax,df.index,df["Inflation"].min(),df["Inflation"].max())
ax.set_title("YoY Inflation Rate (%)  [CPI pct_change(12)]",
             fontweight="bold",color=C["blue"])
ax.legend(fontsize=8)

# 4d Unemployment
ax = fig.add_subplot(gs[1,1])
ax.plot(df.index, df["Unemployment"], color=C["gold"], lw=1.8)
ax.fill_between(df.index, df["Unemployment"], where=rec,
                alpha=0.25, color=C["red"], label="NBER Recession")
annotate_events(ax,df.index,df["Unemployment"].min(),df["Unemployment"].max())
ax.set_title("Unemployment Rate (%)", fontweight="bold", color=C["blue"])
ax.legend(fontsize=8)

# 4e Interest Rate
ax = fig.add_subplot(gs[1,2])
ax.plot(df.index, df["InterestRate"], color=C["purple"], lw=1.8)
ax.fill_between(df.index, df["InterestRate"], where=rec,
                alpha=0.2, color=C["red"], label="NBER Recession")
annotate_events(ax,df.index,df["InterestRate"].min(),df["InterestRate"].max())
ax.set_title("Fed Funds Rate (%)", fontweight="bold", color=C["blue"])
ax.legend(fontsize=8)

# 4f GDP growth
ax = fig.add_subplot(gs[2,0])
ax.bar(df.index, df["GDP_growth"]*100,
       color=[C["red"] if v<0 else C["teal"] for v in df["GDP_growth"]],
       width=20, alpha=0.7)
ax.axhline(0, color=C["text"], lw=1)
ax.fill_between(df.index,-5,5,where=rec,alpha=0.12,color=C["red"])
ax.set_title("Monthly GDP Growth (%)\nRed bars = contraction", fontweight="bold",color=C["blue"])

# 4g Inflation rolling vol
ax = fig.add_subplot(gs[2,1])
ax.plot(df.index, df["Infl_vol12"], color=C["teal"], lw=1.8)
ax.fill_between(df.index,df["Infl_vol12"],where=rec,alpha=0.25,color=C["red"])
ax.axhline(df["Infl_vol12"].mean(),color=C["gold"],ls="--",lw=1.5,
           label=f"Mean {df['Infl_vol12'].mean():.2f}")
ax.set_title("Inflation Volatility (12M rolling std)",
             fontweight="bold",color=C["blue"])
ax.legend(fontsize=8)

# 4h 12-month Fed Funds Rate change
ax = fig.add_subplot(gs[2,2])
ax.plot(df.index, df["Rate_12M_Change"], color=C["purple"], lw=1.8)
ax.fill_between(df.index, 0, df["Rate_12M_Change"],
                where=df["Rate_12M_Change"]<0, alpha=0.25, color=C["red"],
                label="Rate falling (easing)")
ax.axhline(0,color=C["text"],lw=1,ls="--")
ax.fill_between(df.index,-15,15,where=rec,alpha=0.1,color=C["red"])
ax.set_title("12-Month Change in Fed Funds Rate\n⚠ Rate-of-change proxy — NOT a yield-curve spread",
             fontweight="bold",color=C["blue"])
ax.legend(fontsize=8)

# 4i Normalised Z-score
ax = fig.add_subplot(gs[3,:])
for col,color in zip(["GDP","Inflation","Unemployment","InterestRate"],
                     [C["blue"],C["red"],C["gold"],C["purple"]]):
    z = (df[col]-df[col].mean())/df[col].std()
    ax.plot(df.index, z, label=col, color=color, lw=1.3, alpha=0.85)
ax.fill_between(df.index,-4,4,where=rec,alpha=0.1,color=C["red"],label="NBER Recession")
ax.axhline(0,color=C["text"],lw=0.7,alpha=0.5)
ax.set_title(f"All Indicators Z-Score — NBER Recessions Shaded ({rec_months} months, {len(NBER_RECESSIONS)} episodes)",
             fontweight="bold",color=C["blue"])
ax.set_ylabel("Z-Score"); ax.legend(ncol=5,fontsize=8,loc="upper left")

plt.savefig(f"{OUTPUT_DIR}/macro_01_eda.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_01_eda.png")


# =========================================================
# 5. CORRELATION
# =========================================================
print("\n  5. CORRELATION")
core = ["GDP","Inflation","Unemployment","InterestRate",
        "GDP_growth","Infl_change","Rate_change","Unemp_change",
        "Infl_vol12","Rate_vol12","Rate_12M_Change","Recession"]
corr = df[core].corr(numeric_only=True)

fig,axes = plt.subplots(1,2,figsize=(18,7))
fig.suptitle(f"Correlation Analysis — NBER Recessions ({rec_months} months)",
             fontsize=14,fontweight="bold",color=C["blue"])
mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,ax=axes[0],mask=mask,annot=True,fmt=".2f",cmap="RdBu_r",
            center=0,linewidths=0.5,linecolor="white",annot_kws={"size":7},
            cbar_kws={"shrink":0.8})
axes[0].set_title("Correlation Matrix\n(NBER recession label, YoY inflation)",
                  fontweight="bold",color=C["blue"])
axes[0].tick_params(axis="x",rotation=45,labelsize=7)

rec_corr = corr["Recession"].drop("Recession").sort_values()
axes[1].barh(rec_corr.index, rec_corr.values,
             color=[C["red"] if v>0 else C["green"] for v in rec_corr],
             edgecolor="white")
axes[1].axvline(0,color=C["text"],lw=0.8)
for i,v in enumerate(rec_corr.values):
    axes[1].text(v+(0.005 if v>=0 else -0.005),i,f"{v:.3f}",va="center",
                 ha="left" if v>=0 else "right",fontsize=8)
axes[1].set_title(f"Correlation with NBER Recession\n({rec_months} months — all {len(NBER_RECESSIONS)} episodes)",
                  fontweight="bold",color=C["blue"])
axes[1].set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_02_correlation.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_02_correlation.png")


# =========================================================
# 6. STATIONARITY (ADF)
# =========================================================
print("\n  6. STATIONARITY — ADF TEST")
fig,ax = plt.subplots(figsize=(10,5))
adf_res=[]
for col in ["GDP","Inflation","Unemployment","InterestRate"]:
    stat,p = adfuller(df[col].dropna())[:2]
    adf_res.append({"Series":col,"ADF":round(stat,3),"p":round(p,4),
                    "Result":"Stationary ✓" if p<0.05 else "Non-stationary ✗"})
adf_df = pd.DataFrame(adf_res)
print(adf_df.to_string(index=False))
colors_a = [C["green"] if p<0.05 else C["red"] for p in adf_df["p"]]
ax.bar(adf_df["Series"],adf_df["p"],color=colors_a,edgecolor="white",width=0.55)
ax.axhline(0.05,color=C["gold"],ls="--",lw=2,label="p=0.05 threshold")
for i,(p,r) in enumerate(zip(adf_df["p"],adf_df["Result"])):
    ax.text(i,p+0.002,r,ha="center",fontsize=8,
            color=C["green"] if p<0.05 else C["red"],fontweight="bold")
ax.set_title(f"ADF Stationarity Test — {len(df):,} monthly obs\n"
             "Non-stationary series should be differenced before modelling",
             fontweight="bold",color=C["blue"])
ax.set_ylabel("ADF p-value")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_03_stationarity.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_03_stationarity.png")


# =========================================================
# 7. RECESSION PREDICTION ML
#    • No leakage: only lag features used
#    • Class imbalance: sample_weight for Gradient Boosting
#    • Walk-forward validation: TimeSeriesSplit (5 folds)
# =========================================================
print("\n  7. RECESSION PREDICTION ML")
print("     [Walk-forward CV | No feature leakage | Class-imbalance weights]")

# Only lagged / rolling features — zero contemporaneous leakage
lag_cols = [c for c in df.columns
            if any(s in c for s in ["_lag1","_lag3","_lag6","MA12","MA6",
                                     "vol12","_change","Rate_12M"])
            and c != "Recession"]

X = df[lag_cols]
y = df["Recession"]

split = int(len(X)*0.8)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

print(f"  Training rows : {len(X_tr)}  Recession months in train: {y_tr.sum()} ({y_tr.mean():.1%})")
print(f"  Test rows     : {len(X_te)}  Recession months in test : {y_te.sum()} ({y_te.mean():.1%})")

sc = StandardScaler()
Xtr_sc = sc.fit_transform(X_tr)
Xte_sc = sc.transform(X_te)

# Walk-forward CV (TimeSeriesSplit — 5 expanding-window folds)
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
print(f"\n  Walk-forward CV: TimeSeriesSplit(n_splits={N_SPLITS})")
print("  Each fold trains on past data only → no future look-ahead")

# Sample weights for class imbalance (Gradient Boosting)
sw_tr = compute_sample_weight("balanced", y_tr)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42),
    "Gradient Boost": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=42),
        # NOTE: GradientBoostingClassifier has no class_weight param →
        #       we pass sample_weight to fit() for imbalance correction
}

results = {}
for name, model in models.items():
    # Walk-forward cross-validation
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(Xtr_sc)):
        Xf_tr, Xf_val = Xtr_sc[tr_idx], Xtr_sc[val_idx]
        yf_tr, yf_val = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        sw_fold = compute_sample_weight("balanced", yf_tr)
        if name == "Gradient Boost":
            model.fit(Xf_tr, yf_tr, sample_weight=sw_fold)
        else:
            model.fit(Xf_tr, yf_tr)
        if yf_val.nunique() > 1:
            fold_auc = roc_auc_score(yf_val, model.predict_proba(Xf_val)[:,1])
            cv_aucs.append(fold_auc)
        else:
            pass

    # Final fit on full training set
    if name == "Gradient Boost":
        model.fit(Xtr_sc, y_tr, sample_weight=sw_tr)
    else:
        model.fit(Xtr_sc, y_tr)

    yp = model.predict_proba(Xte_sc)[:,1]
    yd = model.predict(Xte_sc)
    auc = roc_auc_score(y_te, yp) if y_te.nunique() > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y_te, yp)
    rep = classification_report(y_te, yd, output_dict=True, zero_division=0)
    cv_arr = np.array(cv_aucs)
    results[name] = {
        "model": model, "yp": yp, "yd": yd, "auc": auc,
        "fpr": fpr, "tpr": tpr,
        "cv_mean": np.nanmean(cv_arr), "cv_std": np.nanstd(cv_arr),
        "recall": rep.get("1",{}).get("recall",0),
        "f1":     rep.get("1",{}).get("f1-score",0),
    }
    print(f"\n  {name:<20} Test-AUC={auc:.4f}  "
          f"WF-CV AUC={np.nanmean(cv_arr):.4f}±{np.nanstd(cv_arr):.4f}  "
          f"Recall={results[name]['recall']:.3f}  F1={results[name]['f1']:.3f}")

best = max(results, key=lambda k: results[k]["auc"])
print(f"\n  ★ Best model: {best}")

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(16,13))
fig.suptitle(
    "Recession Prediction ML — NBER Labels | Walk-Forward CV | Class-Imbalance Weights\n"
    "⚠ All features lagged ≥1 month (no look-ahead leakage). "
    f"CV: TimeSeriesSplit (5 folds). Imbalance: sample_weight=balanced.",
    fontsize=11, fontweight="bold", color=C["blue"]
)

mc = {"Random Forest":C["teal"],"Gradient Boost":C["red"]}
for name, res in results.items():
    cv_lbl = f"WF-CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}"
    axes[0,0].plot(res["fpr"], res["tpr"], lw=2.5, color=mc[name],
                   label=f"{name}  AUC={res['auc']:.3f}  {cv_lbl}")
axes[0,0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random")
axes[0,0].set_xlabel("FPR"); axes[0,0].set_ylabel("TPR")
axes[0,0].set_title("ROC Curves — Test Set\n(WF-CV AUC = walk-forward cross-validation)",
                    fontweight="bold",color=C["blue"])
axes[0,0].legend(fontsize=8)

cm = confusion_matrix(y_te, results[best]["yd"])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
labels = [[f"{v}\n({p:.0f}%)" for v,p in zip(r,rp)] for r,rp in zip(cm,cm_pct)]
sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=axes[0,1],
            xticklabels=["Expansion","Recession"],
            yticklabels=["Expansion","Recession"],
            linewidths=1, linecolor="white", annot_kws={"size":11})
axes[0,1].set_title(f"Confusion Matrix — {best}\n"
                    f"(class_weight=balanced / sample_weight applied)",
                    fontweight="bold", color=C["blue"])
axes[0,1].set_xlabel("Predicted"); axes[0,1].set_ylabel("Actual")

fi = pd.Series(results[best]["model"].feature_importances_,
               index=lag_cols).sort_values().tail(15)
axes[1,0].barh(fi.index, fi.values,
               color=[C["red"] if v>fi.quantile(0.75) else C["teal"] for v in fi.values],
               edgecolor="white")
axes[1,0].set_title(f"Top 15 Feature Importances ({best})\n"
                    "(lagged features only — no contemporary leakage)",
                    fontweight="bold",color=C["blue"])
axes[1,0].set_xlabel("Importance")

test_dates = df.index[split:]
axes[1,1].plot(test_dates, results[best]["yp"], color=C["red"], lw=1.8,
               label="Recession probability")
axes[1,1].fill_between(test_dates, 0.5, 1,
                        where=results[best]["yp"]>0.5, alpha=0.2, color=C["red"])
axes[1,1].axhline(0.5, color=C["gold"], ls="--", lw=1.5, label="Threshold 0.5")
act = y_te.values==1
if act.any():
    axes[1,1].scatter(test_dates[act], results[best]["yp"][act],
                      color=C["red"], zorder=5, s=60, label="Actual recession")
annotate_events(axes[1,1], test_dates, 0, 1)
axes[1,1].set_ylabel("P(Recession)"); axes[1,1].legend(fontsize=8)
axes[1,1].set_title("Predicted Recession Probability (Test Period)",
                    fontweight="bold", color=C["blue"])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_04_recession_ml.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_04_recession_ml.png")


# =========================================================
# 8. GDP FORECAST — RIDGE + NAÏVE PERSISTENCE BENCHMARK
# =========================================================
print("\n  8. GDP FORECAST")
fdf = df.copy()
fdf["GDP_target"] = fdf["GDP"].shift(-1)
fdf.dropna(inplace=True)

# Leakage-safe feature set: only lagged values of GDP and covariates
gdp_feat = [
    "GDP_lag1","GDP_lag3","GDP_lag6",
    "Inflation_lag1","Unemployment_lag1","InterestRate_lag1",
    "GDP_MA12","Infl_MA6","Rate_change","Unemp_change"
]
# GDP_lag1 is GDP at t-1, target is GDP at t+1 → 2-step ahead, no leakage ✓

Xf = fdf[gdp_feat]; yf = fdf["GDP_target"]
sf = int(len(Xf) * 0.8)
Xf_tr, Xf_te = Xf.iloc[:sf], Xf.iloc[sf:]
yf_tr, yf_te = yf.iloc[:sf], yf.iloc[sf:]

scg = StandardScaler()
Xf_tr_s = scg.fit_transform(Xf_tr)
Xf_te_s = scg.transform(Xf_te)

# Ridge regression — walk-forward CV on training set
lr = Ridge(alpha=1.0)
lr.fit(Xf_tr_s, yf_tr)
pred = lr.predict(Xf_te_s)

# Naïve persistence benchmark: predict GDP(t+1) = GDP(t)
# GDP_lag1 at position sf is the contemporaneous GDP just before the test window
naive_pred = fdf["GDP_lag1"].iloc[sf:].values  # lag1 = GDP at t → predicts t+1

mae   = mean_absolute_error(yf_te, pred)
rmse  = np.sqrt(mean_squared_error(yf_te, pred))
r2    = r2_score(yf_te, pred)
mape  = np.mean(np.abs((yf_te.values - pred) / yf_te.values)) * 100

mae_n  = mean_absolute_error(yf_te, naive_pred)
rmse_n = np.sqrt(mean_squared_error(yf_te, naive_pred))
r2_n   = r2_score(yf_te, naive_pred)
mape_n = np.mean(np.abs((yf_te.values - naive_pred) / yf_te.values)) * 100

print(f"  Ridge   MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
print(f"  Naïve   MAE={mae_n:,.0f}  RMSE={rmse_n:,.0f}  R²={r2_n:.4f}  MAPE={mape_n:.2f}%")


fig, axes = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle(
    "GDP Forecast — Ridge Regression vs Naïve Benchmark",
    fontsize=11, fontweight="bold", color=C["blue"]
)

dates_te = Xf.index[sf:]

axes[0,0].plot(dates_te, yf_te.values, color=C["blue"], lw=2, label="Actual")
axes[0,0].plot(dates_te, pred, color=C["red"], lw=2, ls="--", label=f"Ridge  MAPE={mape:.2f}%")
axes[0,0].plot(dates_te, naive_pred, color=C["gold"], lw=1.5, ls=":", label=f"Naïve  MAPE={mape_n:.2f}%")
axes[0,0].fill_between(dates_te, yf_te.values, pred, alpha=0.12, color=C["red"])
annotate_events(axes[0,0], dates_te, yf_te.min(), yf_te.max())
axes[0,0].set_title("Actual vs Ridge vs Naïve Persistence", fontweight="bold", color=C["blue"])
axes[0,0].legend(fontsize=9)

res_ridge = yf_te.values - pred
res_naive = yf_te.values - naive_pred
axes[0,1].bar(dates_te, res_ridge, color=[C["red"] if r<0 else C["green"] for r in res_ridge],
              alpha=0.6, width=20, label="Ridge residuals")
axes[0,1].plot(dates_te, res_naive, color=C["gold"], lw=1.4, alpha=0.8, label="Naïve residuals")
axes[0,1].axhline(0, color=C["text"], lw=1)
axes[0,1].set_title("Residuals — Ridge (bars) vs Naïve (line)", fontweight="bold", color=C["blue"])
axes[0,1].legend(fontsize=8)

mn, mx = min(yf_te.min(), pred.min()), max(yf_te.max(), pred.max())
axes[1,0].scatter(yf_te.values, pred, alpha=0.4, color=C["teal"], s=18, edgecolors="white",
                  label=f"Ridge  R²={r2:.4f}")
axes[1,0].scatter(yf_te.values, naive_pred, alpha=0.2, color=C["gold"], s=12,
                  label=f"Naïve  R²={r2_n:.4f}")
axes[1,0].plot([mn,mx],[mn,mx],"k--",lw=1.5)
axes[1,0].set_title("Scatter: Actual vs Predicted", fontweight="bold", color=C["blue"])
axes[1,0].set_xlabel("Actual GDP"); axes[1,0].set_ylabel("Predicted GDP")
axes[1,0].legend(fontsize=9)

coef = pd.Series(lr.coef_, index=gdp_feat).sort_values()
axes[1,1].barh(coef.index, coef.values,
               color=[C["red"] if v>0 else C["green"] for v in coef.values], edgecolor="white")
axes[1,1].axvline(0, color=C["text"], lw=1)


plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_05_gdp_forecast.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_05_gdp_forecast.png")


# =========================================================
# 9. ARIMA(2,1,2)
# =========================================================
print("\n  9. ARIMA FORECAST")
gdp_s = df["GDP"]
gdp_d = gdp_s.diff().dropna()
adf_lv = adfuller(gdp_s)[1]; adf_d1 = adfuller(gdp_d)[1]
print(f"  ADF levels: p={adf_lv:.4f}  |  first diff: p={adf_d1:.4f}")


n_hold = max(12, int(len(gdp_s)*0.10))
tr_ar = gdp_s.iloc[:-n_hold]; te_ar = gdp_s.iloc[-n_hold:]
print(f"  Train: {len(tr_ar)}  Hold-out: {n_hold} months")

arm  = ARIMA(tr_ar, order=(2,1,2)).fit()
fobj = arm.get_forecast(steps=n_hold)
fval = fobj.predicted_mean; fci = fobj.conf_int()
fi_idx = pd.date_range(start=tr_ar.index[-1], periods=n_hold+1, freq="ME")[1:]
fval.index = fi_idx; fci.index = fi_idx

# Naïve ARIMA benchmark: last observed value
naive_arima = np.full(n_hold, tr_ar.iloc[-1])

mae_a  = mean_absolute_error(te_ar.values, fval.values)
mape_a = np.mean(np.abs((te_ar.values - fval.values) / te_ar.values)) * 100
mae_an = mean_absolute_error(te_ar.values, naive_arima)
mape_an= np.mean(np.abs((te_ar.values - naive_arima) / te_ar.values)) * 100
print(f"  ARIMA(2,1,2)  MAE={mae_a:,.0f}  MAPE={mape_a:.2f}%")
print(f"  Naïve (last val)                  MAE={mae_an:,.0f}  MAPE={mape_an:.2f}%")

fig, axes = plt.subplots(2, 2, figsize=(16,11))
fig.suptitle(
    f"GDP ARIMA(2,1,2) Forecast — d=1 (ADF p={adf_d1:.3f})\n"
    f"Hold-out={n_hold} months | Naïve benchmark shown | "
    "⚠ Single split (walk-forward CV not applied to ARIMA)",
    fontsize=11, fontweight="bold", color=C["blue"]
)

ax = axes[0,0]
ax.plot(gdp_s.index, gdp_s, color=C["blue"], lw=1.6, label="Historical")
ax.plot(fval.index, fval, color=C["red"], lw=2, ls="--", label="ARIMA Forecast")
ax.fill_between(fval.index, fci.iloc[:,0], fci.iloc[:,1], alpha=0.2, color=C["red"], label="95% CI")
ax.axvline(tr_ar.index[-1], color=C["gold"], ls="--", lw=1.5, label="Forecast start")
annotate_events(ax, gdp_s.index, gdp_s.min(), gdp_s.max())
ax.set_title("History + ARIMA Forecast", fontweight="bold", color=C["blue"])
ax.set_ylabel("GDP Billions USD"); ax.legend(fontsize=8)

ax = axes[0,1]
ax.plot(te_ar.index, te_ar.values, color=C["blue"], lw=2, label="Actual")
ax.plot(fval.index, fval.values, color=C["red"], lw=2, ls="--",
        label=f"ARIMA  MAPE={mape_a:.2f}%")
ax.plot(fi_idx, naive_arima, color=C["gold"], lw=1.8, ls=":",
        label=f"Naïve  MAPE={mape_an:.2f}%")
ax.fill_between(fval.index, fci.iloc[:,0], fci.iloc[:,1], alpha=0.2, color=C["red"], label="95% CI")
ax.set_title(f"Hold-out Comparison — ARIMA vs Naïve", fontweight="bold", color=C["blue"])
ax.legend(fontsize=8)

plot_acf(gdp_d, lags=30, ax=axes[1,0], color=C["teal"])
axes[1,0].set_title("ACF — Differenced GDP (informs q order)", fontweight="bold", color=C["blue"])
plot_pacf(gdp_d, lags=30, ax=axes[1,1], color=C["red"], method="ywm")
axes[1,1].set_title("PACF — Differenced GDP (informs p order)", fontweight="bold", color=C["blue"])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_06_arima.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_06_arima.png")


# =========================================================
# 10. BUSINESS CYCLE
# =========================================================
print("\n  10. BUSINESS CYCLE")
fig, axes = plt.subplots(2, 2, figsize=(16,11))
fig.suptitle("Business Cycle Analysis — NBER Labels | YoY Inflation",
             fontsize=14, fontweight="bold", color=C["blue"])
sc_c = [C["red"] if r==1 else C["teal"] for r in df["Recession"]]
p1 = mpatches.Patch(color=C["red"], label="Recession")
p2 = mpatches.Patch(color=C["teal"], label="Expansion")

# Phillips Curve
axes[0,0].scatter(df["Unemployment"], df["Inflation"], c=sc_c, alpha=0.3, s=12)
exp = df["Recession"]==0
m,b = np.polyfit(df.loc[exp,"Unemployment"], df.loc[exp,"Inflation"], 1)
xl = np.linspace(df["Unemployment"].min(), df["Unemployment"].max(), 100)
axes[0,0].plot(xl, m*xl+b, color=C["blue"], lw=2, label=f"Expansion slope={m:.2f}")
axes[0,0].axhline(2, color=C["gold"], ls="--", lw=1.2, label="Fed 2% target")
axes[0,0].set_xlabel("Unemployment (%)"); axes[0,0].set_ylabel("YoY Inflation (%)")
axes[0,0].set_title("Phillips Curve — 70 Years\n(YoY inflation vs unemployment)",
                    fontweight="bold", color=C["blue"])
axes[0,0].legend(handles=[p1, p2, axes[0,0].lines[0], axes[0,0].lines[1]], fontsize=8)

# GDP growth distribution
ge = df[df["Recession"]==0]["GDP_growth"]*100
gr = df[df["Recession"]==1]["GDP_growth"]*100
axes[0,1].hist(ge, bins=40, alpha=0.6, color=C["teal"],
               label=f"Expansion  μ={ge.mean():.2f}%", edgecolor="white")
axes[0,1].hist(gr, bins=20, alpha=0.6, color=C["red"],
               label=f"Recession  μ={gr.mean():.2f}%", edgecolor="white")
axes[0,1].axvline(0, color=C["text"], lw=1.5, ls="--")
axes[0,1].set_xlabel("GDP Growth (%)"); axes[0,1].set_ylabel("Frequency")
axes[0,1].set_title(f"GDP Growth Distribution\n(NBER: {df['Recession'].sum()} recession months)",
                    fontweight="bold", color=C["blue"])
axes[0,1].legend(fontsize=9)

# Inflation boxplot by phase
bp = axes[1,0].boxplot(
    [df[df["Recession"]==0]["Inflation"].dropna(),
     df[df["Recession"]==1]["Inflation"].dropna()],
    labels=["Expansion","Recession"], patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor(C["teal"]); bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor(C["red"]);  bp["boxes"][1].set_alpha(0.6)
axes[1,0].axhline(2, color=C["gold"], ls="--", lw=1.5, label="Fed 2% target")
axes[1,0].set_ylabel("YoY Inflation (%)"); axes[1,0].legend()
axes[1,0].set_title("Inflation by Business Cycle Phase\n(YoY % — not raw CPI index)",
                    fontweight="bold", color=C["blue"])

# Interest rate path before recession onset
starts = [df.index[i] for i in range(1,len(df))
          if df["Recession"].iloc[i]==1 and df["Recession"].iloc[i-1]==0]
windows = []
for st in starts:
    w = df.loc[:st,"InterestRate"].iloc[-13:-1]
    if len(w)==12: windows.append(w.values)
if windows:
    avg = np.mean(windows, axis=0); std = np.std(windows, axis=0)
    x = np.arange(-12, 0)
    axes[1,1].plot(x, avg, color=C["purple"], lw=2.5, label=f"Avg ({len(windows)} episodes)")
    axes[1,1].fill_between(x, avg-std, avg+std, alpha=0.2, color=C["purple"], label="±1 std")
    axes[1,1].axvline(0, color=C["red"], ls="--", lw=1.5, label="Recession onset")
    axes[1,1].set_xlabel("Months before recession"); axes[1,1].set_ylabel("Fed Funds Rate (%)")
    axes[1,1].set_title(f"Rate Path Into Recessions ({len(windows)} NBER episodes)",
                        fontweight="bold", color=C["blue"])
    axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macro_07_business_cycle.png",dpi=140,bbox_inches="tight",facecolor="white")
plt.close()
print("  ✓ Saved: macro_07_business_cycle.png")


# =========================================================
# 11. SUMMARY
# =========================================================
print("\n"+"="*65)
print("  FINAL SUMMARY")
print("="*65)
print(f"""
  📊 DATASET
     Obs          : {len(df):,} monthly  {df.index[0].date()} → {df.index[-1].date()}
     Recession    : {df['Recession'].sum()} months ({df['Recession'].mean():.1%}) — NBER official
     Episodes     : {len(NBER_RECESSIONS)} NBER recessions

  🔧 METHODS USED
     Feature Engineering
     EDA
     Machine Learning
     Time Series Forecasting
     Walk-Forward Validation

  🤖 RECESSION MODEL ({best})
     Test AUC    : {results[best]['auc']:.4f}
     WF-CV AUC  : {results[best]['cv_mean']:.4f} ± {results[best]['cv_std']:.4f}
     Recall      : {results[best]['recall']:.4f}
     F1          : {results[best]['f1']:.4f}

  📈 GDP FORECAST — RIDGE
     R²          : {r2:.4f}
     MAPE        : {mape:.2f}%
     Naïve MAPE  : {mape_n:.2f}%   (persistence benchmark)
     

  📉 ARIMA(2,1,2)
     MAPE        : {mape_a:.2f}%
     Naïve MAPE  : {mape_an:.2f}%   (last-observation benchmark)

  📁 OUTPUTS  (saved to ./{OUTPUT_DIR}/)
     macro_01_eda.png
     macro_02_correlation.png
     macro_03_stationarity.png
     macro_04_recession_ml.png
     macro_05_gdp_forecast.png
     macro_06_arima.png
     macro_07_business_cycle.png

  ⚠  LIMITATIONS REMINDER
    GDP monthly values are interpolated.
    Models are for educational purposes.
""")
print("="*65)