import pandas as pd
import numpy as np
import scipy.stats as st
from typing import Dict


RENAME_DICT: Dict[str, str] = {
    'generation biomass': 'gen_bio', 
    'generation fossil brown coal/lignite': 'gen_lig', 
    'generation fossil coal-derived gas': 'gen_coal_gas', 
    'generation fossil gas': 'gen_gas', 
    'generation fossil hard coal': 'gen_coal', 
    'generation fossil oil': 'gen_oil', 
    'generation fossil oil shale': 'gen_oil_shale', 
    'generation fossil peat': 'gen_peat', 
    'generation geothermal': 'gen_geo', 
    'generation hydro pumped storage consumption': 'gen_hyd_pump', 
    'generation hydro run-of-river and poundage': 'gen_hyd_river', 
    'generation hydro water reservoir': 'gen_hyd_res', 
    'generation marine': 'gen_mar', 
    'generation nuclear': 'gen_nuc', 
    'generation other': 'gen_other', 
    'generation other renewable': 'gen_oth_renew', 
    'generation solar': 'gen_sol', 
    'generation waste': 'gen_waste', 
    'generation wind offshore': 'gen_wind_off', 
    'generation wind onshore': 'gen_wind_on', 
    'total load actual': 'load_actual', 
    'price day ahead': 'price_dayahead', 
    'price actual': 'price'
}


def clean_energy_df(energy_df: pd.DataFrame) -> pd.DataFrame:
    # Prepare idx
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True, infer_datetime_format=True).dt.tz_localize(None)
    energy_df.set_index('time', inplace=True)

    # Drop columns with all null values
    energy_df.dropna(axis=1, how="all", inplace=True)

    # Interpolate missing values from remaining rows
    energy_df = energy_df.interpolate(method ="bfill")

    # Drop columns with all values equal to 0
    energy_df = energy_df.loc[:, (energy_df!=0).any(axis=0)]

    # Drop "forecast" columns
    energy_df = energy_df.drop(energy_df.filter(regex="forecast").columns, axis=1, errors="ignore")

    # Rename columns
    energy_df.rename(columns=RENAME_DICT, inplace=True)

    # Remove duplicated rows
    energy_df = energy_df.loc[~energy_df.index.duplicated()]

    # Remove duplicated columns
    energy_df = energy_df.loc[:, ~energy_df.columns.duplicated(keep='first')]

    # Calculate the absolute z_scores
    z_scores = np.abs(st.zscore(energy_df))

    # Remove values where z_score is over 2.5 stdev
    for col in energy_df.columns:
        energy_df.loc[z_scores[col] > 2.5] = np.nan

    # Replace outliers with mean/median values for that column
    means_dict = {col: energy_df[col].median() for col in energy_df.columns}
    energy_df.fillna(value=means_dict, inplace=True)

    return energy_df


def clean_weather_df(weather_df: pd.DataFrame) -> pd.DataFrame:
    # Prepare idx
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'], utc=True, infer_datetime_format=True).dt.tz_localize(None)
    weather_df.set_index('dt_iso', inplace=True)

    # Drop columns with all values equal to 0
    weather_df = weather_df.loc[:, (weather_df!=0).any(axis=0)]

    # drop unnecessary columns
    drop_cols = ["rain_3h", "weather_id", "weather_main", "weather_description", "weather_icon"]
    weather_df.drop(drop_cols, inplace=True, axis=1, errors="ignore")

    # temperature: kelvin to celsius
    temp_cols = [col for col in weather_df.columns if "temp" in col]
    weather_df[temp_cols] = weather_df[temp_cols].filter(like="temp").applymap(lambda t: t - 273.15)

    # convert int and float64 columns to float32
    intcols = list(weather_df.dtypes[weather_df.dtypes == np.int64].index)
    weather_df[intcols] = weather_df[intcols].applymap(np.float32)

    f64cols = list(weather_df.dtypes[weather_df.dtypes == np.float64].index)
    weather_df[f64cols] = weather_df[f64cols].applymap(np.float32)

    f32cols = list(weather_df.dtypes[weather_df.dtypes == np.float32].index)

    # Remove duplicated rows
    weather_df = (
        weather_df
        .reset_index()
        .drop_duplicates(subset=["dt_iso", "city_name"], keep="first")
        .set_index('dt_iso')
    )

    # Remove duplicated columns
    weather_df = weather_df.loc[:, ~weather_df.columns.duplicated(keep='first')]

    # Group & concatenate weather DF
    gb_df = weather_df.groupby("city_name")

    def format_df(df: pd.DataFrame, city: str):
        df.drop(columns=['city_name'], inplace=True)
        return df.add_suffix(f"_{city.replace(' ', '')}")

    concat_weather_df: pd.DataFrame = pd.concat(
        [format_df(gb_df.get_group(city), city) for city in gb_df.groups.keys()],
        axis=1
    ).interpolate()

    # Calculate the absolute z_scores
    num_cols = list(concat_weather_df.select_dtypes(include=['number']).columns)
    z_scores = np.abs(st.zscore(concat_weather_df[num_cols]))

    # Remove values where z_score is over 2.5 stdev
    for col in num_cols:
        concat_weather_df.loc[z_scores[col] > 2.5] = np.nan

    # Replace outliers with mean values for that column
    means_dict = {col: concat_weather_df[col].mean() for col in num_cols}
    concat_weather_df.fillna(value=means_dict, inplace=True)

    return concat_weather_df