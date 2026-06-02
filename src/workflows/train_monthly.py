import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.monthly_model.monthly_climademic_model import ClimademicMonthlyModel

_columns_list=['t2m_q_0.75', 'd2m_q_0.75', 'tp_q_0.75', 'si10_q_0.75',
               'Population_Density', 'land_use_0', 'land_use_11',
               'land_use_22', 'land_use_33', 'land_use_44', 'land_use_55',
               'land_use_66', 'land_use_77']

#TODO for all functions define type check
def calculate_quantiles(dataframe, parameters=['t2m', 'd2m', 'tp', 'si10'], quantile_value=0.75):
    quantile_df = dataframe.copy()

    for param in parameters:
        quntile_col_name = param + "_q_" + str(quantile_value)
        col_range = [f"{param}_{i}" for i in range(1, 13)]
        quantile_df[quntile_col_name] = quantile_df[col_range].quantile(quantile_value,axis=1)
    return quantile_df

def load_training_data(training_data_path, specie):
    df_specie = pd.read_csv(training_data_path,low_memory=False)
    df_specie = df_specie.loc[df_specie["VECTOR"] == specie]
    return df_specie.copy()

def prepare_training_dataset(dataframe):
    df = dataframe[_columns_list].copy()
    df = df.dropna(axis=0)

    X = df.copy()
    y = np.ones(len(X))

    X_train, _, y_train, _  = train_test_split(X, y, test_size=0.2, random_state=2) #TODO enable the test_size as parameter
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    save_scaler(scaler, Path(models_dir, specie, "scaler"))

    return X_train_scaled.tolist(), y_train.tolist(), scaler

def save_scaler(scaler, path):
    scaler_path = Path(path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)


def get_gmod_file(gmod_dir, year) :
    gmod_file= next(Path(gmod_dir).glob(f"{year}_*"), None)

    if gmod_file is None:
        raise FileNotFoundError(
            f"No GMOD file found for year {year} in directory: {gmod_dir}"
        )

    return gmod_file

def prepare_gmod_dataset(gmod_file):
    gmod_df = pd.read_csv(gmod_file)
    gmod_df = gmod_df.rename(
        columns={
            "t2m": "t2m_q_0.75",
            "d2m": "d2m_q_0.75",
            "tp": "tp_q_0.75",
            "si10": "si10_q_0.75",
        }
    )

    gmod_df = gmod_df.drop(columns=["Unnamed: 0"], errors="ignore")
    gmod_df = gmod_df.dropna()

    #metadata = gmod_df[["latitude", "longitude", "geometry"]].copy()
    features = gmod_df[ _columns_list]

    #return metadata, features
    return  features


def predict_gmod_probabilities(model, scaler, features):
    X_scaled = scaler.transform(features).tolist()

    predicted_labels, _, predicted_values = model.predict(X_scaled, probability=True)

    probabilities = pd.DataFrame(predicted_values, columns=["Prob_1", "Prob_2"])
    probabilities["P_Label"] = predicted_labels

    return np.array(predicted_values)

def update_training_datasets(X_train, y_train, features,
                          predicted_probabilities, scaler, threshold: float = 0.5):

    missclassified_indices = np.where(predicted_probabilities[:, 0] < threshold)[0]

    if len(missclassified_indices) > 0:
        missclassified_features = features.loc[missclassified_indices]
        missclassified_features_scaled = scaler.transform(missclassified_features).tolist()

        X_train.extend(missclassified_features_scaled)
        y_train.extend(np.ones(len(missclassified_features_scaled), dtype=int).tolist())

    return X_train, y_train





fname = "/Users/MalyshevaN-Dev/Documents/aedes_model/Climademic_Suit_Model/artifacts/ts/mosquito_climate_pop_land_use_n.csv"
genus = "Aedes"
specie = "aegypti"
vector = f"{genus} {specie}"

models_dir = "/Users/MalyshevaN-Dev/Documents/aedes_model/Climademic_Suit_Model/artifacts/models/"
gmod_dir = Path("/Users/MalyshevaN-Dev/Documents/aedes_model/Climademic_Suit_Model/artifacts/processed_data/GMOD_dataset/")

df =load_training_data(fname, vector)

df = calculate_quantiles(df)

X_train, y_train, scaler  = prepare_training_dataset(df)

model = ClimademicMonthlyModel(params="-s 2 -t 2 -g 0.03 -n 0.03 -b 1")
model.train(X_train, y_train)

model_name = f"{specie}/base_ocsvm_{specie}_quantile_75.model'"
model_path = Path(models_dir, model_name)
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_path)

years = np.arange(2015, 2016)

for year in years:

    gmod_filename = get_gmod_file(Path(gmod_dir), year)

    features = prepare_gmod_dataset(gmod_filename)

    gmod_prob = predict_gmod_probabilities(model, scaler, features)

    X_train, y_train = update_training_datasets(X_train, y_train, features,
                          gmod_prob, scaler)
    
    model.train(X_train, y_train)
    model_name = f"{specie}/{year}_ocsvm_{specie}_quantile_75.model'"
    model_path = Path(models_dir, model_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)








#print(df.head(5))