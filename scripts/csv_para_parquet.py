import pandas as pd
import os

def csv_para_parquet(path_csv, path_final="../data"):
    df = pd.read_csv(path_csv)

    if not os.path.exists(path_final):
        os.makedirs(path_final)

    path_arquivo_parquet = os.path.join(path_final, os.path.splitext(os.path.basename(path_csv))[0] + ".parquet")
    
    df.to_parquet(path_arquivo_parquet, engine="pyarrow")
    print(f'Arquivo salvo em: {path_arquivo_parquet}')


csv_para_parquet("../data/dblp-v10.csv")