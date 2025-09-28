"""
Titanic – Klasifikacija
Autor: Ognjen Ikrašev RA 238/2022
Opis:
    - Učitava train.csv (Kaggle Titanic format)
    - Radi EDA (grafici i tabele), obradu nedostajućih vrednosti, inženjering atributa
    - Enkodira kategorijske atribute preko OneHotEncoder-a
    - Trenira i poredi više modela (LogisticRegression, RandomForest, GradientBoosting)
    - Radi k-unakrsnu validaciju sa metrikama: accuracy, precision, recall, f1
    - Vizuelizuje: distribucije, preživljavanje po atributima, značajnost atributa (feature importances & permutation importance)
    - Upoređuje model sa svim atributima vs. samo najbitnijim atributima
    - Kreira PNG grafikone

Upotreba:
    1) Postavite train.csv u isti folder.
    2) Pokrenite:  python Titanic_Klasifikacija.py
    3) Rezultati će se sačuvati u ./outputs/ (grafici, tabele)

"""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif



warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 120, "figure.autolayout": True})

# folderi za rezultate i pravljenje ako ne postoje
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

# klasa za cuvanje rezultata svakog modela
@dataclass
class ModelResult:
    name: str                       # naziv modela
    metrics: Dict[str, float]       # metrike (f1, accuracy...)
    fitted_pipeline: Pipeline       # ceo pipeline sa preprocessing + modelom
    feature_names: List[str]        # imena atributa nakon enkodovanja

# funkcija za ucitavanje podataka
def load_data(path: str = "train.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nije pronađen fajl '{path}'. Postavite train.csv u radni direktorijum."
        )
    df = pd.read_csv(path)
    return df

# izvlacimo titulu jer moze biti jako korisna varijabla za klasifikaciju
# iz titule mozemo znati pol, starosnu grupu kao i drustveni status
# gde sve tri informacije mogu znacajno uticati na to da li ce biti izabrani za spasavanje
def extract_title(name: str) -> str:
    # Izvlačenje titule iz imena (npr. "Mr.", "Mrs.", "Master.")
    if pd.isna(name):
        return "Unknown"
    match = re.search(r",\s*([^\.]+)\.", name)
    title = match.group(1).strip() if match else "Unknown"
    # Grupisanje retkih titula
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Rare"

# izvlacenje prefiksa karte ako postoji
# moze da donose malo poboljsanje tacnosti modela
def ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket):
        return "UNKNOWN"
    t = ticket.replace(".", "").replace("/", "").upper()
    parts = [p for p in t.split() if not p.isdigit()]
    return parts[0] if parts else "NUMERIC"

# vraca slovo palube, vraca U - unknown ako nema
# moze biti korisno kao pokazatelj vrednosti putnika
def cabin_deck(cabin: str) -> str:
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin) == 0:
        return "U"
    return cabin[0].upper()

# iznenjering novih atributa sa kojima ce modeli lakse raditi
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Izvedene promenljive
    df["Title"] = df["Name"].apply(extract_title)       # titula iz imena
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["TicketPrefix"] = df["Ticket"].apply(ticket_prefix)
    df["CabinDeck"] = df["Cabin"].apply(cabin_deck)

    # podela po godinama
    age_bins = [0, 12, 18, 25, 35, 50, 65, 120]
    df["AgeBin"] = pd.cut(df["Age"], bins=age_bins,
                          labels=["child", "teen", "yadult", "adult", "mature", "senior", "elder"], include_lowest=True)

    # Cena karte po log-skali (stabilizuje raspodelu)
    df["FareLog"] = np.log1p(df["Fare"].fillna(0))

    return df       # vracanje novih podataka


def basic_eda_plots(df: pd.DataFrame, save: bool = True) -> None:
    # 1) Raspodela starosti (hist)
    plt.figure()
    df["Age"].dropna().plot(kind="hist", bins=30)
    plt.title("Raspodela starosti (Age)")
    plt.xlabel("Godine")
    if save:
        plt.savefig(os.path.join(FIG_DIR, "raspodela_starosti.png"))
    plt.close()

    # 2) Preživljavanje po polu
    plt.figure()
    df["Sex_label"] = df["Sex"].map({"female": "zena", "male": "muskaraca"})
    df["Survived_label"] = df["Survived"].map({0: "Nisu preživeli", 1: "Preživeli"})
    pd.crosstab(df["Sex_label"], df["Survived_label"]).plot(kind="bar", stacked=True)
    plt.title("Preživljavanje po polu (Sex)")
    plt.xlabel("Pol")
    plt.ylabel("Broj putnika")
    if save:
        plt.savefig(os.path.join(FIG_DIR, "prezivljanje_po_polu.png"))
    plt.close()

    # 3) Preživljavanje po klasi
    plt.figure()
    df["Survived_label"] = df["Survived"].map({0: "Nisu preživeli", 1: "Preživeli"})
    pd.crosstab(df["Pclass"], df["Survived_label"]).plot(kind="bar", stacked=True)
    plt.title("Preživljavanje po klasi (Pclass)")
    plt.xlabel("Klasa")
    plt.ylabel("Broj putnika")
    if save:
        plt.savefig(os.path.join(FIG_DIR, "prezivljavanje_po_klasi.png"))
    plt.close()

    # 4) Preživljavanje po luci ukrcavanja
    plt.figure()
    df["Survived_label"] = df["Survived"].map({0: "Nisu preživeli", 1: "Preživeli"})
    pd.crosstab(df["Embarked"], df["Survived_label"]).plot(kind="bar", stacked=True)
    plt.title("Preživljavanje po luci (Embarked)")
    plt.xlabel("Luka ukrcavanja")
    plt.ylabel("Broj putnika")
    if save:
        plt.savefig(os.path.join(FIG_DIR, "prezivljavanje_po_luci.png"))
    plt.close()


def build_preprocessor(feature_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # Definišemo koje kolone odbacujemo (neinformativne / ID, duplikati informacije itd.)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]

    # Numeričke i kategorijske kolone nakon inženjeringa
    num_cols = [c for c in feature_df.columns if c not in drop_cols]
    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "TicketPrefix", "CabinDeck", "AgeBin"]

    # privremeno podela – OneHot će raditi i nad binovima
    num_base = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FareLog"]
    num_cols = [c for c in num_cols if c in num_base]

    # Numerički pipeline (popuni nedostajuće vrednosti sa medijalnom)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Kategorijski pipeline (popuni najčešćim + OneHotEncoder)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # kombinovanje u jedan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols

# funkcija za kros-validaciju modela
def evaluate_models(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer) -> List[ModelResult]:
    # pravljenje modela i ubacivanje u listu
    models = [
        ("LogReg", LogisticRegression(max_iter=1000, n_jobs=None, solver="lbfgs")),
        ("RF", RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42)),
        ("GBoost", GradientBoostingClassifier(random_state=42)),
    ]

    # smestanje u rezultat
    results: List[ModelResult] = []

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    # StratifiedKFold → deli podatke tako da razmera klasa (0/1) ostane ista
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # radim kros-validaciju za sve modele i racunam prosecne rezultate
    for name, clf in models:
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_estimator=True)     # kros-validacija
        metrics = {m: float(np.mean(cv[f"test_{m}"])) for m in scoring.keys()}      # racunanje rezultata

        # Fitovanje celog modela da ga mogu posle analizirati
        pipe.fit(X, y)

        # Dohvati imena kolona posle OneHotEncoder-a
        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        feature_names = pipe.named_steps["pre"].get_feature_names_out()

        # get_feature_names_out vrati num i cat exp naziv; uprosti imena
        feature_names = [fn.replace("num__", "").replace("cat__", "") for fn in feature_names]

        # stavljam rezultate u ModelResult
        results.append(ModelResult(name=name, metrics=metrics, fitted_pipeline=pipe, feature_names=feature_names))

        # cuvanje tabele metrika po foldovima
        fold_table = pd.DataFrame({
            "fold": list(range(1, len(cv["fit_time"]) + 1)),
            "accuracy": cv["test_accuracy"],
            "precision": cv["test_precision"],
            "recall": cv["test_recall"],
            "f1": cv["test_f1"],
        })
        fold_table.to_csv(os.path.join(TABLE_DIR, f"cv_folds_{name}.csv"), index=False)

    return results


def plot_feature_importance(model_res: ModelResult, X: pd.DataFrame, y: pd.Series, top_k: int = 20) -> pd.DataFrame:
    # Uzimamo trenirani pipeline i sam klasifikator (clf) iz njega
    pipe = model_res.fitted_pipeline
    clf = pipe.named_steps["clf"]

    # 1. Nativna važnost atributa (feature_importances_ ili coef_)
    # vec postoji u modelima
    importances = None
    if hasattr(clf, "feature_importances_"):
        # RandomForest, GradientBoosting – oni imaju .feature_importances_
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # LogisticRegression – koeficijenti se koriste kao mera značaja
        coefs = clf.coef_.ravel()
        # normalizujemo da mozemo uporediti
        importances = np.abs(coefs) / (np.linalg.norm(coefs) + 1e-9)

    if importances is not None:
        # Ako imamo native importances, pravimo DataFrame
        n = min(len(model_res.feature_names), len(importances))
        fi_df = pd.DataFrame({"feature": model_res.feature_names, "importance": importances})
        # Sortiramo i uzimamo top_k najvažnijih
        fi_df = fi_df.sort_values("importance", ascending=False).head(top_k)

        # Crtamo horizontalni graf
        plt.figure()
        plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])     # [::-1] da najveći bude na vrhu
        plt.title(f"{model_res.name} – Značajnost atributa (native)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"feature_importance_native_{model_res.name}.png"))
        plt.close()
    else:
        # ako nema nativnih importanci vracamo prazan DataFrame
        fi_df = pd.DataFrame()

    # Permutation importance (stabilnije tumačenje)
    # proveravamo svaki feature koliko je bitan za model
    # Radimo na train split-u (sa test velicinom od 25%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    # ponovo fitujemo na train podacima
    pipe.fit(X_train, y_train)

    # Računamo permutation importance: permutujemo atribute i gledamo F1 score-a
    perm = permutation_importance(pipe, X_val, y_val, n_repeats=15, random_state=42, scoring="f1")

    # Uzimamo stvarna imena feature-a iz X
    feature_names = X.columns

    # Pravimo DataFrame koji je uvek iste dužine
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    })

    # Sortiranje i uzimanje top_k
    perm_df = perm_df.sort_values(by="importance", ascending=False).head(top_k)

    # crtamo permutation graf za najbolji model
    plt.figure()
    plt.barh(perm_df["feature"][::-1], perm_df["importance"][::-1])
    plt.title(f"{model_res.name} – Permutation importance (F1)")
    plt.xlabel("ΔF1 pri permutaciji")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"feature_importance_permutation_{model_res.name}.png"))
    plt.close()

    return perm_df

# gledamo koji feature je najbitniji
def select_top_features(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
    # uzimamo imena kolona iz preprocessora
    preprocessor.fit(X)     # Fitujemo preprocesor da sazna sve transformacije
    feat_names = preprocessor.get_feature_names_out()       # imena novih kolona
    # uklanjamo prefikse kako bi ocistili imena
    feat_names = [fn.replace("num__", "").replace("cat__", "") for fn in feat_names]

    # Transformiši X da bi računali MI (zahteva numeričke podatke)
    X_enc = preprocessor.transform(X)
    # mutual_info_classif meri koliko informacija svaka kolona "nosi" o target-u y
    mi = mutual_info_classif(X_enc, y, random_state=42)

    # rangiranje i izbor top k atributa
    order = np.argsort(mi)[::-1]    # indexi od najinformativnijeg do najmanje inf
    # uzimamo prvih k elemenata i njihova imena
    top_idx = order[: min(k, len(order))]
    top_features = [feat_names[i] for i in top_idx]

    # Sačuvaj tabelu MI
    mi_df = pd.DataFrame({"feature": feat_names, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_df.to_csv(os.path.join(TABLE_DIR, "mutual_information.csv"), index=False)

    return top_features


def train_with_selected_features(model_name: str, base_pre: ColumnTransformer, X: pd.DataFrame, y: pd.Series,
                                 selected_features: List[str]) -> ModelResult:
    # Funkcija trenira model samo na unapred izabranim kolonama (npr. top 20 po MI).
    # 1. Odabere samo tražene kolone iz originalnog DataFrame-a
    # 2. Kreira novi ColumnTransformer koji radi scaling na numeričkim i OneHot kodiranje na kategorijskim kolonama
    # 3. Treniramo model
    # 4. Vraćamo rezultate (cross-validation metrike, finalni model, imena feature-a)

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.base import BaseEstimator, TransformerMixin

    class ColumnSelectorByName(BaseEstimator, TransformerMixin):
        """
        Custom transformer koji bira kolone po imenu iz DataFrame-a.
        Radi u sklearn pipeline-u.
        """

        def __init__(self, columns):
            self.columns = columns  # lista imena kolona koje cuvamo

        def fit(self, X, y=None):
            # pronalazimo indekse kolona u DataFrame-u i čuvamo ih
            if isinstance(X, pd.DataFrame):
                self._indices = [X.columns.get_loc(col) for col in self.columns]
            else:
                raise ValueError("ColumnSelectorByName očekuje pd.DataFrame kao ulaz")
            return self

        def transform(self, X):
            # koristimo zapamćene indekse da izdvojimo kolone
            if not hasattr(self, "_indices"):
                raise AttributeError("Morate prvo pozvati fit pre transform")
            return X.iloc[:, self._indices].copy()


    # Fitujemo originalni preprocessor da izvučemo sva imena feature-a
    pre = base_pre
    pre.fit(X)

    # razdvojimo numeričke i kategorijske kolone
    num_feats = [f for f in selected_features if X[f].dtype in [int, float]]
    cat_feats = [f for f in selected_features if X[f].dtype == object]

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    # biramo koji cemo model da realizujemo (klasifikator)
    if model_name == "RF":
        clf = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Model {model_name} nije podrzan")

    # kompletan pipeline (preprocesor + klasifikator)
    pipe = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])

    # stratifikovani k-fold
    # koristimo jer imamo veliki skup podataka i daje nam pouzdanije treniranje modela
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # metrike za evaluaciju, kros-validacija i prosek svih metrika preko foldova
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_estimator=True)
    metrics = {m: float(np.mean(cv[f"test_{m}"])) for m in scoring.keys()}

    # Fit final
    pipe.fit(X, y)

    # Pošto posle selekcije gubimo originalna imena, dodelićemo samo izabrana
    model_res = ModelResult(name=f"RF_top{len(selected_features)}", metrics=metrics, fitted_pipeline=pipe,
                            feature_names=selected_features)

    # Sačuvaj fold metrike
    fold_table = pd.DataFrame({
        "fold": list(range(1, len(cv["fit_time"]) + 1)),
        "accuracy": cv["test_accuracy"],
        "precision": cv["test_precision"],
        "recall": cv["test_recall"],
        "f1": cv["test_f1"],
    })
    fold_table.to_csv(os.path.join(TABLE_DIR, f"cv_folds_{model_res.name}.csv"), index=False)

    return model_res


def main():
    # 1) Učitavanje i inženjering
    df = load_data("train.csv")
    df = engineer_features(df)

    # 2) EDA grafici
    basic_eda_plots(df)

    # 3) Priprema X, y
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # 4) Evaluacija različitih modela (CV)
    results = evaluate_models(X, y, preprocessor)

    # Izbor najboljeg po F1 (možete promeniti kriterijum)
    best = max(results, key=lambda r: r.metrics["f1"])

    # 5) Značajnost atributa – nativna + permutaciona
    perm_df = plot_feature_importance(best, X, y, top_k=20)
    perm_df.to_csv(os.path.join(TABLE_DIR, f"top_permutation_importance_{best.name}.csv"), index=False)

    # 6) Selekcija najbitnijih atributa po MI i poređenje performansi
    top_feats = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]

    res_top = train_with_selected_features("RF", preprocessor, X, y, top_feats)
    results.append(res_top)

    # 7) Sačuvaj pregled metrika
    metrics_table = pd.DataFrame([
        {"model": r.name, **r.metrics} for r in results
    ]).sort_values("f1", ascending=False)
    metrics_table.to_csv(os.path.join(TABLE_DIR, "model_metrics.csv"), index=False)

    # 9) Kratak ispis u konzoli
    print("\n===== Metrike po modelima (5-fold CV) =====")
    for r in results:
        print(
            f"{r.name:10s} | acc={r.metrics['accuracy']:.3f}  prec={r.metrics['precision']:.3f}  rec={r.metrics['recall']:.3f}  f1={r.metrics['f1']:.3f}")
    print("\nGrafici i tabele sačuvani su u ./outputs/")


main()
