# Titanic Passenger EDA — Riya Pathak

## Project objective
Perform basic Exploratory Data Analysis (EDA) on the Titanic `train.csv` dataset using Python, NumPy, Pandas and Matplotlib. This is EDA only: **no machine-learning model is built**.

## Dataset
Download **only `train.csv`** from the Kaggle Titanic competition:
https://www.kaggle.com/competitions/titanic/data

After downloading, put the file here:

```text
Titanic_Passenger_EDA_Riya_Pathak/
└── data/
    └── train.csv
```

The assignment specifies 891 rows in `train.csv`. The program reads the file and prints its actual shape so you can verify it.

## Project structure

```text
Titanic_Passenger_EDA_Riya_Pathak/
├── data/
│   └── train.csv          # Add the Kaggle file here
├── outputs/               # Charts and summary CSVs are saved here
├── main.py                # Complete EDA program
├── requirements.txt
└── README.md
```

## Run in VS Code

1. Install Python and the Python extension in VS Code.
2. Open the folder `Titanic_Passenger_EDA_Riya_Pathak` in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (recommended):

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install required libraries:
   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Download `train.csv` from Kaggle and place it in the `data` folder.
7. Run:
   ```powershell
   python main.py
   ```

## What the program covers

- Displays `head()`, `tail()`, `shape`, `info()` and `describe()`.
- Checks missing values and duplicate rows.
- Counts survivors and non-survivors.
- Counts passengers by sex and passenger class.
- Calculates average age and fare using NumPy arrays.
- Compares survival counts by sex and passenger class using Pandas crosstabs and `groupby()`.
- Reports missing values in Age, Cabin and Embarked.
- Lists the 10 highest fares using Pandas sorting/filtering.
- Creates a survival-count bar chart and an age histogram.
- Prints three observations based on the dataset.
- Saves both charts and supporting summary CSV files in `outputs/`.

## Notes
- `Survived`: 0 = did not survive; 1 = survived.
- `Pclass`: 1 = first class, 2 = second class, 3 = third class.
- Missing Age values are excluded from the average-age calculation and age histogram; no values are silently imputed.
- Cabin is included in the missing-value check, as required.
- The code uses the Kaggle training file only; it does not load `test.csv` or `gender_submission.csv`.
