def prepare_features(df):
    selected_features = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'NAME_EDUCATION_TYPE',
    ]

    X = df[selected_features].copy()

    # How many rows have this anomalous value?
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)

    # DAYS_BIRTH is negative: -X means the person was born X days before the application
    # Dividing by -365.25 converts to positive years (365.25 accounts for leap years)
    X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365.25).round(1)

    # DAYS_EMPLOYED works the same way
    X['YEARS_EMPLOYED'] = (-X['DAYS_EMPLOYED'] / 365.25).round(1)

    # DAYS_ID_PUBLISH: years since the ID document was issued
    X['YEARS_ID_PUBLISH'] = (-X['DAYS_ID_PUBLISH'] / 365.25).round(1)

    X = X.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH'])

    # Encode categorical variables
    # For simple binary variables, we can use 0/1 encoding. 
    X['CODE_GENDER'] = X['CODE_GENDER'].map({'M': 0, 'F': 1}).fillna(0).astype(int)
    X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1}).astype(int)
    X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1}).astype(int)

    # For education, we use ordinal encoding
    education_map = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4,
    }
    X['EDUCATION_LEVEL'] = X['NAME_EDUCATION_TYPE'].map(education_map).fillna(1).astype(int)
    X = X.drop(columns=['NAME_EDUCATION_TYPE'])

    # Keep missing values here.
    # The training pipeline will learn imputation values from training data only.


    # Feature Engineering
    # Credit-to-income ratio: how many years of income does the loan represent?
    # A ratio of 5 means the loan is 5x the annual income — that's a heavy burden
    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)

    # Annuity-to-income ratio: what fraction of income goes to monthly payments?
    # A ratio of 0.3 means 30% of income goes to loan payments — that's stressful
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)

    # Credit-to-goods ratio: how much of the goods price is financed?
    # A ratio close to 1 means the person is financing nearly 100% of the purchase
    # (no down payment), which is a risk signal
    X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)
    return X