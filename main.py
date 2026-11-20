"""
Titanic Passenger EDA
Student: Riya Pathak

This project performs basic Exploratory Data Analysis (EDA) on Kaggle's
Titanic train.csv dataset. No machine-learning model is built.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # 1. Load only train.csv
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "train.csv was not found. Download train.csv from Kaggle and "
            "place it inside the project's data folder."
        )

    df = pd.read_csv(DATA_FILE)

    # Check that this is the expected Titanic training dataset
    required_columns = {
        "Survived", "Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print("=" * 65)
    print("TITANIC PASSENGER EDA | RIYA PATHAK")
    print("=" * 65)

    # 2. Basic dataset exploration
    print("\n1. FIRST FIVE ROWS")
    print(df.head())

    print("\n2. LAST FIVE ROWS")
    print(df.tail())

    print("\n3. DATASET SHAPE (rows, columns)")
    print(df.shape)

    print("\n4. DATASET INFORMATION")
    df.info()

    print("\n5. NUMERICAL SUMMARY")
    print(df.describe())

    # 3. Missing values and duplicate records
    print("\n6. MISSING VALUES IN EACH COLUMN")
    print(df.isnull().sum())

    print("\n7. TOTAL DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n8. MISSING VALUES IN REQUIRED COLUMNS")
    print(df[["Age", "Cabin", "Embarked"]].isnull().sum())

    # 4. Question 1: survivors and non-survivors
    survival_counts = df["Survived"].value_counts().reindex([0, 1], fill_value=0)
    print("\n9. SURVIVORS AND NON-SURVIVORS")
    print(f"Non-survivors (0): {survival_counts[0]}")
    print(f"Survivors (1):     {survival_counts[1]}")

    # 5. Question 2: passenger counts by sex and passenger class
    print("\n10. PASSENGER COUNT BY SEX")
    print(df["Sex"].value_counts())

    print("\n11. PASSENGER COUNT BY PASSENGER CLASS")
    print(df["Pclass"].value_counts().sort_index())

    print("\n12. PASSENGER COUNT BY SEX AND PASSENGER CLASS")
    print(pd.crosstab(df["Sex"], df["Pclass"]))

    # 6. Question 3: average age and fare (NumPy used for numeric calculations)
    age_array = df["Age"].dropna().to_numpy()
    fare_array = df["Fare"].dropna().to_numpy()

    print("\n13. AVERAGE AGE")
    print(f"{np.mean(age_array):.2f} years")

    print("\n14. AVERAGE FARE")
    print(f"{np.mean(fare_array):.2f}")

    print("\n15. NUMPY NUMERIC CHECK")
    print(f"Age array size: {age_array.size}")
    print(f"Fare array size: {fare_array.size}")
    print(f"Median age (NumPy): {np.median(age_array):.2f}")
    print(f"Maximum fare (NumPy): {np.max(fare_array):.2f}")

    # 7. Question 4: survival counts by sex and passenger class
    print("\n16. SURVIVAL COUNTS BY SEX")
    print(pd.crosstab(df["Sex"], df["Survived"], margins=True))

    print("\n17. SURVIVAL COUNTS BY PASSENGER CLASS")
    print(pd.crosstab(df["Pclass"], df["Survived"], margins=True))

    print("\n18. SURVIVAL COUNTS BY SEX AND PASSENGER CLASS")
    print(
        df.groupby(["Sex", "Pclass", "Survived"], observed=False)
          .size()
          .unstack(fill_value=0)
    )

    # 8. Question 5: missing values in Age, Cabin, Embarked
    print("\n19. MISSING VALUES: AGE, CABIN AND EMBARKED")
    print(df[["Age", "Cabin", "Embarked"]].isnull().sum())

    # 9. Question 6: top 10 highest fares
    print("\n20. TOP 10 HIGHEST FARES")
    top_10_fares = df.nlargest(10, "Fare")[["PassengerId", "Name", "Pclass", "Fare"]]
    print(top_10_fares.to_string(index=False))

    # Save useful analysis tables
    top_10_fares.to_csv(OUTPUT_DIR / "top_10_highest_fares.csv", index=False)
    pd.crosstab(df["Sex"], df["Pclass"]).to_csv(OUTPUT_DIR / "passengers_by_sex_and_class.csv")
    pd.crosstab(df["Sex"], df["Survived"]).to_csv(OUTPUT_DIR / "survival_by_sex.csv")
    pd.crosstab(df["Pclass"], df["Survived"]).to_csv(OUTPUT_DIR / "survival_by_class.csv")

    # 10. Chart 1: survival-count bar chart
    plt.figure(figsize=(7, 5))
    labels = ["Did not survive", "Survived"]
    values = [int(survival_counts[0]), int(survival_counts[1])]
    bars = plt.bar(labels, values, color=["#d96b6b", "#4c9f70"])
    plt.title("Titanic Passengers: Survival Count")
    plt.xlabel("Survival Status")
    plt.ylabel("Number of Passengers")
    plt.bar_label(bars, padding=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_count.png", dpi=300)
    plt.show()

    # 11. Chart 2: age histogram (available/non-missing ages)
    plt.figure(figsize=(7, 5))
    plt.hist(age_array, bins=20, color="#5b8db8", edgecolor="black")
    plt.title("Age Distribution of Titanic Passengers")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_histogram.png", dpi=300)
    plt.show()

    # 12. Three observations generated from the dataset
    female_survival = df.loc[df["Sex"] == "female", "Survived"].mean() * 100
    male_survival = df.loc[df["Sex"] == "male", "Survived"].mean() * 100
    class_survival = df.groupby("Pclass")["Survived"].mean() * 100

    print("\n" + "=" * 65)
    print("THREE OBSERVATIONS")
    print("=" * 65)
    print(
        f"1. Female survival rate was {female_survival:.1f}%, while "
        f"male survival rate was {male_survival:.1f}%."
    )
    print(
        f"2. Passenger class 1 had a survival rate of {class_survival.loc[1]:.1f}%, "
        f"class 2 had {class_survival.loc[2]:.1f}%, and "
        f"class 3 had {class_survival.loc[3]:.1f}%."
    )
    print(
        f"3. The average passenger age was {np.mean(age_array):.2f} years "
        f"and the average fare was {np.mean(fare_array):.2f}."
    )

    print("\nCharts and CSV summary tables have been saved in the 'outputs' folder.")
    print("EDA completed. No machine-learning model was built.")

if __name__ == "__main__":
    main()
