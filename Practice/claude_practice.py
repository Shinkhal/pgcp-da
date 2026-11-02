"""
PGCP-AI Data Analytics - practical exam cheat code.
One function per syllabus session. Run one:  python lab_prep.py s05
Run all:  python lab_prep.py
In Jupyter/Colab: paste a function, call it. Replace iris_df() with pd.read_csv('file.csv') in exam.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score, KFold,
                                     StratifiedKFold, GridSearchCV, validation_curve)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_score, recall_score,
                             f1_score, mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)

warnings.filterwarnings("ignore")
sns.set_theme()


# ------------------------------------------------------------------ helpers
def iris_df():
    d = load_iris(as_frame=True)
    df = d.frame.copy()
    df["species"] = pd.Categorical.from_codes(d.target, d.target_names)
    return df


def decide(p, alpha=0.05):
    print(f"p-value = {p:.4g} -> {'REJECT H0' if p < alpha else 'FAIL TO REJECT H0'} (alpha={alpha})")


def messy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, 50).astype(float),
        "income": rng.normal(50000, 12000, 50).round(),
        "city": rng.choice(["Pune", "Delhi", "Blr", None], 50),
    })
    df.loc[rng.choice(50, 6, replace=False), "age"] = np.nan
    return pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 duplicate rows


# ------------------------------------------------------------ S1-2 lifecycle
def s01_02_lifecycle():
    """Discovery -> prep -> model plan -> build -> QA."""
    # Discovery
    df = iris_df()
    print(df.shape); df.info(); print(df.head()); print(df.describe(include="all"))
    # Data prep on messy data
    m = messy()
    print("missing:\n", m.isna().sum(), "\ndups:", m.duplicated().sum())
    m = m.drop_duplicates()
    m["age"] = m["age"].fillna(m["age"].median())
    m["city"] = m["city"].fillna(m["city"].mode()[0])
    m = pd.get_dummies(m, columns=["city"], drop_first=True)
    m[["age", "income"]] = MinMaxScaler().fit_transform(m[["age", "income"]])
    print(m.head())
    # sklearn imputer alternative
    imp = SimpleImputer(strategy="median")
    print(imp.fit_transform(messy()[["age", "income"]])[:3])
    # Model planning/build/QA
    X, y = df.drop(columns=["target", "species"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(Xtr, ytr)
    print("test acc:", accuracy_score(yte, pipe.predict(Xte)))
    print("cv acc  :", cross_val_score(pipe, X, y, cv=5).mean())


# ---------------------------------------------------------- S3 nature of data
def s03_nature_of_data():
    df = iris_df()
    print(df.dtypes)
    num, cat = df.select_dtypes("number"), df.select_dtypes(exclude="number")
    print("numeric:", list(num), "| categorical:", list(cat))
    print(df.nunique())
    print("missing %:\n", df.isna().mean() * 100)
    print("skew:\n", num.skew())
    print("kurt:\n", num.kurt())
    print(df["species"].value_counts(normalize=True))
    print(df.groupby("species", observed=True).agg(["mean", "std"]).T.head(8))


# ---------------------------------------------------------------- S4 visualize
def s04_visualization():
    df = iris_df()
    col = "sepal length (cm)"
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    sns.histplot(df[col], kde=True, ax=ax[0, 0])
    sns.boxplot(data=df, x="species", y=col, ax=ax[0, 1])
    sns.violinplot(data=df, x="species", y="petal length (cm)", ax=ax[0, 2])
    sns.scatterplot(data=df, x=col, y="petal length (cm)", hue="species", ax=ax[1, 0])
    sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm", ax=ax[1, 1])
    sns.countplot(data=df, x="species", ax=ax[1, 2])
    plt.tight_layout(); plt.show()
    sns.pairplot(df.drop(columns="target"), hue="species"); plt.show()
    # insight sample
    print(df.groupby("species", observed=True)["petal length (cm)"].mean())


# ------------------------------------------------------------ S5 descriptive
def s05_descriptive():
    df = iris_df()
    x = df["sepal length (cm)"]
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
    std, var = x.std(ddof=1), x.var(ddof=1)
    print(f"mean={x.mean():.3f} median={x.median():.3f} mode={x.mode()[0]}")
    print(f"Q1={q1} Q2={q2} Q3={q3} IQR={q3 - q1:.3f} range={x.max() - x.min():.3f}")
    print(f"P90={np.percentile(x, 90):.3f} std={std:.3f} var={var:.3f} CV={std / x.mean() * 100:.2f}%")
    print("pop std (ddof=0):", x.std(ddof=0))
    print(stats.describe(x))
    print(df.describe().T)
    print(df.groupby("species", observed=True)["sepal length (cm)"].agg(["mean", "median", "std", "var"]))


# ----------------------------------------------------- S6-7 prob + Naive Bayes
def s06_07_probability_naive_bayes():
    df = iris_df()
    col = "sepal length (cm)"
    df["size"] = np.where(df[col] > df[col].median(), "long", "short")
    # sample vs population, resampling
    sample = df.sample(n=30, random_state=1)
    print("pop mean", df[col].mean(), "sample mean", sample[col].mean())
    rng = np.random.default_rng(0)
    boots = [rng.choice(df[col], len(df), replace=True).mean() for _ in range(2000)]
    print("bootstrap 95% CI:", np.percentile(boots, [2.5, 97.5]))
    # joint / marginal / conditional
    joint = pd.crosstab(df["species"], df["size"], normalize="all")
    print("joint\n", joint)
    print("marginal species\n", joint.sum(axis=1))
    print("marginal size\n", joint.sum(axis=0))
    print("P(size|species)\n", pd.crosstab(df["species"], df["size"], normalize="index"))
    # Bayes by hand: P(virginica | long)
    p_long_given_v = ((df["species"] == "virginica") & (df["size"] == "long")).sum() / (df["species"] == "virginica").sum()
    p_v, p_long = (df["species"] == "virginica").mean(), (df["size"] == "long").mean()
    print("Bayes P(virginica|long) =", p_long_given_v * p_v / p_long)
    # medical test: prevalence 1%, sens 99%, spec 95%
    prev, sens, spec = 0.01, 0.99, 0.95
    post = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    print("P(disease|+) =", round(post, 4))
    # Gaussian Naive Bayes
    X, y = df.drop(columns=["target", "species", "size"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    nb = GaussianNB().fit(Xtr, ytr)
    pred = nb.predict(Xte)
    print("acc", accuracy_score(yte, pred)); print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred))
    print("proba first 3:\n", nb.predict_proba(Xte[:3]).round(3))
    # Multinomial NB text (spam)
    msgs = ["win money now", "free prize win", "meeting at noon", "project report due", "free money offer", "lunch tomorrow"]
    lab = [1, 1, 0, 0, 1, 0]
    cv = CountVectorizer(); Xt = cv.fit_transform(msgs)
    mnb = MultinomialNB().fit(Xt, lab)
    print("spam?", mnb.predict(cv.transform(["free money", "see you at the meeting"])))


# ------------------------------------------------------------ S8 distributions
def s08_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, 1000)
    print("shapiro", stats.shapiro(x)); decide(stats.shapiro(x).pvalue)
    print("D'Agostino", stats.normaltest(x))
    print("KS", stats.kstest((x - x.mean()) / x.std(), "norm"))
    print("skew", stats.skew(x), "kurt", stats.kurtosis(x))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(x, kde=True, ax=ax[0]); stats.probplot(x, plot=ax[1]); plt.show()
    # discrete / continuous
    print("Binom P(X=3), n=10,p=.5:", stats.binom.pmf(3, 10, 0.5), "P(X<=3):", stats.binom.cdf(3, 10, 0.5))
    print("Poisson P(X=2), lam=3:", stats.poisson.pmf(2, 3), "P(X>=5):", stats.poisson.sf(4, 3))
    print("Normal P(X<60):", stats.norm.cdf(60, 50, 10), "95th pct:", stats.norm.ppf(0.95, 50, 10))
    # CLT: means of skewed population approach normal
    pop = rng.exponential(2, 100000)
    means = rng.choice(pop, (2000, 30)).mean(axis=1)
    print("pop mean/std", pop.mean(), pop.std(), "| means mean/std", means.mean(), means.std(), "theory SE", pop.std() / np.sqrt(30))
    print("means normal? p =", stats.shapiro(means).pvalue)
    sns.histplot(means, kde=True); plt.title("CLT n=30"); plt.show()


# ------------------------------------------------------ S9 sampling/estimation
def s09_sampling_estimation():
    df = iris_df()
    col = "sepal length (cm)"
    rng = np.random.default_rng(1)
    simple = df.sample(n=30, random_state=1)
    strat = df.groupby("species", observed=True).sample(n=10, random_state=1)
    k = len(df) // 30
    start = rng.integers(0, k)
    systematic = df.iloc[start::k]
    df["cluster"] = np.arange(len(df)) // 15
    cl = rng.choice(df["cluster"].unique(), 3, replace=False)
    cluster = df[df["cluster"].isin(cl)]
    for name, s in [("simple", simple), ("stratified", strat), ("systematic", systematic), ("cluster", cluster)]:
        print(f"{name:11s} n={len(s):3d} mean={s[col].mean():.3f}")
    s = simple[col]
    se = stats.sem(s)
    print("SE", se, "| 95% CI (t):", stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=se))
    z = stats.norm.ppf(0.975)
    print("95% CI (z):", s.mean() - z * s.std(ddof=1) / np.sqrt(len(s)), s.mean() + z * s.std(ddof=1) / np.sqrt(len(s)))
    # required sample size for margin E, sigma: n = (z*sigma/E)^2
    print("n for E=0.1:", int(np.ceil((z * df[col].std() / 0.1) ** 2)))


# ------------------------------------------------- S10 correlation + outliers
def s10_correlation_outliers():
    df = iris_df()
    a, b = "sepal length (cm)", "petal length (cm)"
    print("cov matrix\n", df[[a, b]].cov()); print("np.cov", np.cov(df[a], df[b])[0, 1])
    r, p = stats.pearsonr(df[a], df[b])
    print(f"pearson r={r:.4f} p={p:.3g}", "| spearman", stats.spearmanr(df[a], df[b])[0])
    print("strength:", "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.3 else "weak", "positive" if r > 0 else "negative")
    # inject outliers
    d = df.copy(); d.loc[[0, 1, 2], a] = [25, 30, -5]
    x = d[a]
    z = np.abs(stats.zscore(x)); print("z>3 idx:", list(x.index[z > 3]))
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print("IQR fences", lo, hi, "outliers:", list(x.index[(x < lo) | (x > hi)]))
    removed = d[(d[a] >= lo) & (d[a] <= hi)]                       # 1 drop
    capped = d.assign(**{a: x.clip(lo, hi)})                      # 2 cap / winsorize
    med = d.assign(**{a: x.where((x >= lo) & (x <= hi), x.median())})  # 3 replace by median
    print("shapes", removed.shape, capped.shape, med.shape)
    print("corr before", d[[a, b]].corr().iloc[0, 1], "after removal", removed[[a, b]].corr().iloc[0, 1])
    sns.boxplot(x=x); plt.show()


# ---------------------------------------------- S11-12 hypothesis testing
def s11_12_hypothesis_testing():
    from statsmodels.stats.weightstats import ztest
    df = iris_df()
    col = "sepal length (cm)"
    x = df[col]
    # one-sample Z (large n or known sigma): H0 mu=6
    z, p = ztest(x, value=6); print("Z-test", z, end=" "); decide(p)
    z_manual = (x.mean() - 6) / (x.std(ddof=1) / np.sqrt(len(x))); print("manual z", z_manual)
    # two-sample z / t
    a, b = df[df.species == "setosa"][col], df[df.species == "virginica"][col]
    print("2-sample z", ztest(a, b)); print("Welch t", stats.ttest_ind(a, b, equal_var=False)); decide(stats.ttest_ind(a, b, equal_var=False).pvalue)
    print("1-sample t", stats.ttest_1samp(x, 6))
    print("paired t", stats.ttest_rel(df["sepal length (cm)"], df["petal length (cm)"]))
    # ANOVA across 3 species
    print("ANOVA", stats.f_oneway(*[g[col] for _, g in df.groupby("species", observed=True)]))
    # chi-square independence
    df["size"] = np.where(x > x.median(), "long", "short")
    tab = pd.crosstab(df["species"], df["size"])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    print(tab); print(f"chi2={chi2:.3f} dof={dof}", end=" "); decide(p); print("expected\n", exp.round(2))
    # chi-square goodness of fit (die fair?)
    print("GoF", stats.chisquare([18, 22, 16, 14, 12, 18], [100 / 6] * 6))
    # skewness
    print("skew", stats.skew(x), "| skewtest", stats.skewtest(x), "| kurtosis", stats.kurtosis(x))
    print("interpretation: |skew|<0.5 ~symmetric; >0 right tail; <0 left tail")


# -------------------------------------- S13-16 predictive & prescriptive models
def s13_16_trees_predictive_prescriptive():
    df = iris_df()
    X, y = df.drop(columns=["target", "species"]), df["target"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    # information gain by hand
    def entropy(v):
        p = np.bincount(v) / len(v); p = p[p > 0]
        return -(p * np.log2(p)).sum()
    def info_gain(x, v, thr):
        l, r = v[x <= thr], v[x > thr]
        return entropy(v) - (len(l) / len(v) * entropy(l) + len(r) / len(v) * entropy(r))
    print("entropy(y)", entropy(ytr))
    best = max(((info_gain(Xtr[c].values, ytr, t), c, t) for c in X.columns for t in np.unique(Xtr[c])), key=lambda z: z[0])
    print("best split (IG, feature, threshold):", best)
    # tree
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(Xtr, ytr)
    print(export_text(clf, feature_names=list(X.columns)))   # trees as set of rules
    print("importance", dict(zip(X.columns, clf.feature_importances_.round(3))))
    print("acc", accuracy_score(yte, clf.predict(Xte)))
    print("class probabilities", clf.predict_proba(Xte[:3]).round(2))
    plt.figure(figsize=(12, 6)); plot_tree(clf, feature_names=list(X.columns), class_names=load_iris().target_names, filled=True); plt.show()
    # supervised segmentation by one attribute
    print(df.groupby(pd.cut(df["petal length (cm)"], [0, 2.5, 4.9, 8]), observed=True)["species"].value_counts().unstack())
    # prescriptive: pick action maximizing expected payoff given predicted probabilities
    payoff = np.array([[10, -2, -2],    # action A pays per true class
                       [-2, 8, 1],      # action B
                       [0, 0, 0]])      # do nothing
    proba = clf.predict_proba(Xte)
    best_action = (proba @ payoff.T).argmax(axis=1)
    print("recommended actions (0=A,1=B,2=none) first 10:", best_action[:10])
    print("Predictive = what will happen; Prescriptive = what to do about it (optimize action on predictions).")


# --------------------------------------------- S17-18 regression + forecasting
def s17_18_regression_forecasting():
    import statsmodels.api as sm
    from sklearn.preprocessing import PolynomialFeatures
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    d = load_diabetes(as_frame=True); X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    # simple regression
    lr = stats.linregress(X["bmi"], y); print("simple: slope", lr.slope, "intercept", lr.intercept, "r2", lr.rvalue ** 2, "p", lr.pvalue)
    # multiple
    m = LinearRegression().fit(Xtr, ytr); pr = m.predict(Xte)
    print("coef", dict(zip(X.columns, m.coef_.round(1))), "b0", m.intercept_)
    print(f"R2={r2_score(yte, pr):.3f} RMSE={mean_squared_error(yte, pr) ** 0.5:.2f} MAE={mean_absolute_error(yte, pr):.2f}")
    print(sm.OLS(ytr, sm.add_constant(Xtr)).fit().summary().tables[1])
    # polynomial / regularized
    poly = Pipeline([("p", PolynomialFeatures(2)), ("s", StandardScaler()), ("r", Ridge(alpha=10))]).fit(Xtr, ytr)
    print("poly+ridge R2", r2_score(yte, poly.predict(Xte)), "| lasso nonzero", (Lasso(alpha=1).fit(Xtr, ytr).coef_ != 0).sum())
    plt.scatter(pr, yte - pr); plt.axhline(0, c="r"); plt.title("residuals"); plt.show()
    # forecasting
    rng = np.random.default_rng(0); t = np.arange(48)
    s = pd.Series(100 + 2 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 48),
                  index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    train, test = s[:36], s[36:]
    print("moving avg(3) last:", s.rolling(3).mean().iloc[-1])
    alpha, ses = 0.3, [train.iloc[0]]
    for v in train.iloc[1:]: ses.append(alpha * v + (1 - alpha) * ses[-1])   # manual SES
    print("SES last level:", ses[-1])
    naive = pd.Series(train.iloc[-1], index=test.index)
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(12)
    ar = ARIMA(train, order=(1, 1, 1)).fit().forecast(12)
    for n, f in [("naive", naive), ("holt-winters", hw), ("arima", ar)]:
        print(f"{n:13s} MAPE={mean_absolute_percentage_error(test, f):.3%} RMSE={mean_squared_error(test, f) ** 0.5:.2f}")
    train.plot(label="train"); test.plot(label="test"); hw.plot(label="HW"); plt.legend(); plt.show()


# ------------------------------------------ S19-20 simulation + optimization
def s19_20_simulation_optimization():
    rng = np.random.default_rng(0)
    # Monte Carlo profit model
    n = 100_000
    demand = rng.normal(1000, 200, n).clip(0)
    price = rng.triangular(18, 20, 24, n)
    profit = demand * (price - 12) - 5000
    print(f"mean={profit.mean():.0f} std={profit.std():.0f} P(loss)={(profit < 0).mean():.3%}")
    print("5/50/95 pct:", np.percentile(profit, [5, 50, 95]).round(0), "| VaR95 (5th pct):", np.percentile(profit, 5).round(0))
    sns.histplot(profit, bins=60); plt.show()
    pts = rng.random((n, 2)); print("pi ~", 4 * ((pts ** 2).sum(axis=1) <= 1).mean())
    # LP: max 3x+5y  s.t. x<=4, 2y<=12, 3x+2y<=18, x,y>=0  -> (2,6), 36
    res = optimize.linprog(c=[-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], bounds=[(0, None)] * 2)
    print("LP:", res.x, "max =", -res.fun, "| status", res.message)
    # integer LP (scipy>=1.9)
    ri = optimize.linprog([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], bounds=[(0, None)] * 2, integrality=[1, 1])
    print("ILP:", ri.x)
    # Nonlinear: min x^2+y^2 s.t. x+y>=1 -> (.5,.5), .5
    nl = optimize.minimize(lambda v: v[0] ** 2 + v[1] ** 2, x0=[2, 2], method="SLSQP",
                           constraints=[{"type": "ineq", "fun": lambda v: v[0] + v[1] - 1}], bounds=[(0, None)] * 2)
    print("NLP:", nl.x.round(4), nl.fun)
    print("1-D:", optimize.minimize_scalar(lambda x: (x - 3) ** 2 + 2).x)


# ------------------------------------------------- S21 overfitting / CV
def s21_overfitting_cv():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    depths = range(1, 16)
    tr, te = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(t.score(Xtr, ytr)); te.append(t.score(Xte, yte))
    print("best depth on holdout:", list(depths)[int(np.argmax(te))], "| gap at depth 15:", tr[-1] - te[-1])
    plt.plot(depths, tr, label="train"); plt.plot(depths, te, label="test"); plt.legend(); plt.xlabel("depth"); plt.show()
    # holdout vs CV
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    print("holdout:", clf.fit(Xtr, ytr).score(Xte, yte))
    print("5-fold :", cross_val_score(clf, X, y, cv=5).round(3), cross_val_score(clf, X, y, cv=5).mean())
    print("strat10:", cross_val_score(clf, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0)).mean())
    print("manual KFold:", np.mean([clf.fit(X[a], y[a]).score(X[b], y[b]) for a, b in KFold(5, shuffle=True, random_state=0).split(X)]))
    gs = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": range(1, 10), "min_samples_leaf": [1, 5, 10]}, cv=5).fit(Xtr, ytr)
    print("grid best:", gs.best_params_, gs.best_score_, "test:", gs.score(Xte, yte))
    trs, vas = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth", param_range=range(1, 11), cv=5)
    print("validation curve mean CV:", vas.mean(axis=1).round(3))


# --------------------------------------------- S22 evaluating classifiers
def s22_evaluate_classifiers():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, pred); tn, fp, fn, tp = cm.ravel()
    print(cm); print(f"acc={accuracy_score(yte, pred):.3f} prec={precision_score(yte, pred):.3f} rec={recall_score(yte, pred):.3f} f1={f1_score(yte, pred):.3f} auc={roc_auc_score(yte, proba):.3f}")
    print("manual precision tp/(tp+fp):", tp / (tp + fp), "recall tp/(tp+fn):", tp / (tp + fn))
    print(classification_report(yte, pred))
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    print("baseline acc (majority):", base.score(Xte, yte), "| model:", model.score(Xte, yte))
    fpr, tpr, _ = roc_curve(yte, proba); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.show()
    # expected value with cost-benefit matrix [[TN,FP],[FN,TP]]
    cb = np.array([[0, -5], [-100, 20]])
    print("expected profit per case:", (cm / cm.sum() * cb).sum())
    # threshold tuning by expected profit
    best = max(((((confusion_matrix(yte, proba >= t) / len(yte)) * cb).sum(), t) for t in np.linspace(.05, .95, 19)))
    print("best threshold (profit, thr):", best)


# --------------------------------------------- S23 evidence combination, Bayes
def s23_bayes_evidence():
    prior = 0.01
    tests = [(0.95, 0.90), (0.90, 0.85), (0.80, 0.95)]    # (sensitivity, specificity) for 3 independent positive tests
    p = prior
    for i, (se, sp) in enumerate(tests, 1):
        p = se * p / (se * p + (1 - sp) * (1 - p))        # posterior becomes next prior
        print(f"after test {i}: P(D) = {p:.4f}")
    odds = prior / (1 - prior)
    for se, sp in tests: odds *= se / (1 - sp)             # naive Bayes: odds x likelihood ratios
    print("odds method:", odds / (1 + odds), "(must equal sequential)")
    # Naive Bayes with explicit evidence: P(spam | words)
    p_spam, lik_spam, lik_ham = 0.4, {"free": .6, "win": .5, "meeting": .02}, {"free": .05, "win": .03, "meeting": .3}
    ev = ["free", "win"]
    ns = p_spam * np.prod([lik_spam[w] for w in ev]); nh = (1 - p_spam) * np.prod([lik_ham[w] for w in ev])
    print("P(spam|free,win) =", ns / (ns + nh))


# ----------------------- S24 factor / directional / functional data analysis
def s24_factor_directional_functional():
    from scipy.signal import savgol_filter
    # Factor analysis
    d = load_wine(as_frame=True); X = StandardScaler().fit_transform(d.data)
    fa = FactorAnalysis(n_components=3, rotation="varimax", random_state=0).fit(X)
    load = pd.DataFrame(fa.components_.T, index=d.data.columns, columns=["F1", "F2", "F3"])
    print(load.round(2)); print("communalities:\n", (load ** 2).sum(axis=1).round(2))
    pca = PCA().fit(X); print("PCA cum var:", pca.explained_variance_ratio_.cumsum().round(2)); print("eigenvalues>1 (Kaiser):", (pca.explained_variance_ > 1).sum())
    plt.plot(range(1, 14), pca.explained_variance_, "o-"); plt.axhline(1, ls="--"); plt.title("scree"); plt.show()
    # Directional (circular) data - never use arithmetic mean on angles
    deg = np.array([350, 10, 5, 355, 20, 340]); rad = np.deg2rad(deg)
    print("naive mean:", deg.mean(), "| circular mean:", np.rad2deg(stats.circmean(rad)) % 360)
    R = np.hypot(np.cos(rad).mean(), np.sin(rad).mean())
    print("resultant length R:", R, "circ var:", 1 - R, "circ std (rad):", stats.circstd(rad))
    # Functional data: curves as functions
    rng = np.random.default_rng(0); t = np.linspace(0, 1, 100)
    curves = np.array([rng.normal(1, .3) * np.sin(2 * np.pi * t) + rng.normal(0, .2) + rng.normal(0, .15, 100) for _ in range(30)])
    smooth = np.array([savgol_filter(c, 15, 3) for c in curves])
    mean_fn = smooth.mean(axis=0); deriv = np.gradient(mean_fn, t)
    fpca = PCA(n_components=3).fit(smooth)
    print("FPCA var ratio:", fpca.explained_variance_ratio_.round(3))
    plt.plot(t, smooth.T, c="grey", alpha=.3); plt.plot(t, mean_fn, "r", lw=3); plt.plot(t, deriv, "b--"); plt.show()


# ------------------------------------------------------------------ S25 notes
# def s25_powerbi_notes():
#     print("""Power BI (no python in exam - GUI). Python visual alt: Get Data > Python script.
# Flow: Get Data > Transform (Power Query) > Close&Apply > Report view.
# Power Query M examples (Advanced Editor):
#   let Source = Csv.Document(File.Contents("C:\\\\data.csv"),[Delimiter=",", Encoding=65001]),
#       Hdr = Table.PromoteHeaders(Source),
#       Typed = Table.TransformColumnTypes(Hdr,{{"Sales", type number}}),
#       NoNull = Table.SelectRows(Typed, each [Sales] <> null),
#       Dedup = Table.Distinct(NoNull)
#   in Dedup
# DAX: Total Sales = SUM(Sales[Amount]) ; YoY = CALCULATE([Total Sales], SAMEPERIODLASTYEAR('Date'[Date]))
# Visuals: bar/line/pie/map/card/slicer; Dashboard = pin visuals; share via Power BI Service publish.""")


SECTIONS = {
    "s01": s01_02_lifecycle, "s03": s03_nature_of_data, "s04": s04_visualization, "s05": s05_descriptive,
    "s06": s06_07_probability_naive_bayes, "s08": s08_distributions, "s09": s09_sampling_estimation,
    "s10": s10_correlation_outliers, "s11": s11_12_hypothesis_testing, "s13": s13_16_trees_predictive_prescriptive,
    "s17": s17_18_regression_forecasting, "s19": s19_20_simulation_optimization, "s21": s21_overfitting_cv,
    "s22": s22_evaluate_classifiers, "s23": s23_bayes_evidence, "s24": s24_factor_directional_functional,
    # "s25": s25_powerbi_notes,
}

if __name__ == "__main__":
    keys = sys.argv[1:] or list(SECTIONS)
    for k in keys:
        print(f"\n{'=' * 20} {k}: {SECTIONS[k].__name__} {'=' * 20}")
        SECTIONS[k]()