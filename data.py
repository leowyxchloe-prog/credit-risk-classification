import kaggle
import pandas as pd

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "tahmidmir/credit-risk-dataset", path="./", unzip=True
)

df = pd.read_excel("Credit_Risk_Data_Set/train-FIN_ANA_DATA .xls")  # sheet_name=0 grabs the first tab
df.to_csv("credit_risk.csv", index=False)  