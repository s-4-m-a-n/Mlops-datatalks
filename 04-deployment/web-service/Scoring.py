
import pickle
import pandas as pd
import click

CATEGORICAL = ['PULocationID', 'DOLocationID']


def load_model(model_file):
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(input_file, model_file, output_file, year, month):
    df = read_data(input_file)
    dv, model = load_model(model_file)
    
    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Mean Prediction: ", y_pred.mean())

    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    df["y_pred"] = y_pred
    output_columns = ["ride_id", "y_pred"]
    df_result = df[output_columns]
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
        )

@click.command()
@click.option('--year', default=1, help='enter year')
@click.option('--month', default=1, help='enter month without 0 in the prefix')
def run(year, month):
    year = int(year)
    month = int(month)

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output-yellow-{year:04d}-{month:02d}.parquet'
    model_file = 'web-service/model.bin'


    apply_model(input_file=input_file,
                model_file=model_file,
                output_file=output_file,
                year=year,
                month=month)



if __name__ == "__main__":
    # year = 2023
    # month = 3
    # run(year, month)
    run()
