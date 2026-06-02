import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.monthly_model.monthly_climademic_model import ClimademicMonthlyModel
from config import CONFIG

_columns_list=['t2m_q_0.75', 'd2m_q_0.75', 'tp_q_0.75', 'si10_q_0.75',
               'pop_density', 'Ocean', 'Urban',
               'Cropland', 'Pasture', 'Forest', 'Shrub',
               'Barren', 'Water']

#TODO for all functions define type check
def calculate_quantiles(dataframe, parameters=['t2m', 'd2m', 'tp', 'si10'], quantile_value=0.75):
    quantile_df = dataframe.copy()

    for param in parameters:
        quntile_col_name = param + "_q_" + str(quantile_value)
        col_range = [f"{param}_{i:02d}" for i in range(1, 13)]
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

    return X_train_scaled.tolist(), y_train.tolist(), scaler

def save_scaler(scaler, path):
    scaler_path = Path(path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)


def get_gmod_file(gmod_dir, year_start, year_end) :

    gmod_file= next(Path(gmod_dir).glob(f"*_{year_start}-{year_end}*"), None)

    if gmod_file is None:
        raise FileNotFoundError(
            f"No GMOD file found for years {year_start}-{year_end} in directory: {gmod_dir}"
        )

    return gmod_file

def prepare_gmod_dataset(gmod_file, specie):
    gmod_df = pd.read_csv(gmod_file)
    gmod_df = gmod_df.loc[gmod_df["Specie"] == specie]
    gmod_df = gmod_df.rename(
        columns={
            "t2m": "t2m_q_0.75",
            "d2m": "d2m_q_0.75",
            "tp": "tp_q_0.75",
            "si10": "si10_q_0.75",
        }
    )
    gmod_df = gmod_df.dropna()
    
    features = gmod_df[ _columns_list + ["year"]]
    return  features


def predict_gmod_probabilities(model, scaler, features):
    X_scaled = scaler.transform(features).tolist()

    predicted_labels, _, predicted_values = model.predict(X_scaled, probability=True)

    probabilities = pd.DataFrame(predicted_values, columns=["Prob_1", "Prob_2"])
    probabilities["P_Label"] = predicted_labels

    return np.array(predicted_values)

def update_training_datasets(X_train, y_train, features,
                                                       predicted_probabilities, scaler, threshold: float = 0.5):

    #define misclassified predictions for given trseshold
    missclassified_indices = np.where(predicted_probabilities[:, 0] < threshold)[0]

    #add misclassified points to th etraining dataset
    if len(missclassified_indices) > 0:
        missclassified_features = features.loc[missclassified_indices]
        missclassified_features_scaled = scaler.transform(missclassified_features).tolist()

        X_train.extend(missclassified_features_scaled)
        y_train.extend(np.ones(len(missclassified_features_scaled), dtype=int).tolist())

    return X_train, y_train

def train_climademic_monthly_model(config, specie, gmod_year_start, gmod_year_end):
    paths = config["paths"]
    vector = f"Aedes {specie}"

    #path for saving trained models
    models_dir = Path(paths["model_parameters"]["models_directory"])
    
    training_df =load_training_data(paths["processed_data"]["training_dataset"], vector)
    training_df_quantiles = calculate_quantiles(training_df)
    
    X_train, y_train, scaler  = prepare_training_dataset(training_df_quantiles)
    save_scaler(scaler, Path(models_dir, specie, "scaler")) #save for future inference
    
    model = ClimademicMonthlyModel(params="-s 2 -t 2 -g 0.03 -n 0.03 -b 1")
    model.train(X_train, y_train)
    
    #save base model (2975-2014)
    model_name = f"{specie}/base_ocsvm_{specie}_quantile_75.model'"
    model_path = Path(models_dir, model_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    
    #iterate over years of incremental learning
    years = np.arange(gmod_year_start, gmod_year_end + 1)
    gmod_filename = get_gmod_file(paths["processed_data"]["gmod_dataset"], gmod_year_start, gmod_year_end)
    features = prepare_gmod_dataset(gmod_filename, vector)

    for year in years:
        #get gmod data for that year
        features_year = features.loc[features["year"] == year].copy()# prepare_gmod_dataset(gmod_filename)
        features_year = features_year.drop(columns=["year"])

        #predict probabilities for given gmod set
        gmod_prob = predict_gmod_probabilities(model, scaler, features_year)

        #add misclassified points to the training dataset
        X_train, y_train = update_training_datasets(X_train, y_train, features_year,
                            gmod_prob, scaler)
        #retrain model
        model.train(X_train, y_train)

        #save  newly trained model
        model_name = f"{specie}/{year}_ocsvm_{specie}_quantile_75.model'"
        model_path = Path(models_dir, model_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)