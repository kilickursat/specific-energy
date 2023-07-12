import io
df = pd.read_excel(io.BytesIO(uploaded['data_wearing.xlsx']))

(df.columns.tolist())

df.isnull().sum()

df2 = df.fillna(0)

df2.isnull().sum()

df2 = df2.drop(['RING NU','Altitude','ExcavationD','pitching','rolling','Middle break left and right fold angle (%)','Middle break upper and lower folds (%)',' Geosanth exploration equipment exploration pressure (kN)',
              'Geoyama Exploration Equipment Exploration Stroke (mm)','Clay shock injection pressure (MPa)','Clay shock flow rateA (L/min)','Clay shock flow rateB (L/min)',
              'Back injection pressure (MPa)','Ef (m3/mm)', ' Rotation angle (degree)','Bubble injection pressure (MPa)','Back in flow rate of A liquid (L/min)','Back in flow rate of B liquid (L/min)','Excavated Tunnel Length (m)',],axis=1)

df2.head()

df2.describe()



import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

"""#Before setting up PyCaret, a random sample of 10% size of the dataset will be get to make predictions with unseen data."""

RANDOM_SEED = 142

def data_sampling(dataset, frac: float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac, random_state=random_seed)
    data_sampled_b =  dataset.drop(data_sampled_a.index).reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b

df2, data_unseen = data_sampling(df2, 0.9, RANDOM_SEED)
print(f"There are {data_unseen.shape[0]} samples for Unseen Data.")

"""#In order to demonstrate the use of the predict_model function on unseen data, a sample of 73 records (~10%) has been withheld from the original dataset to be used for predictions at the end. This should not be confused with a train-test-split."""

print('Data for Modeling: ' + str(df2.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

import matplotlib.pyplot as plt
import seaborn as sns


from pycaret.regression import *

clovrs = ClusterOverSampler(oversampler=SMOTE(random_state=1), clusterer=KMeans(random_state=2), distributor=DensityDistributor(), random_state=3)

Session_2 = setup(df2, target = 'SE (MJ/m^3)', session_id=177, log_experiment=False,
                  experiment_name='specific-energy', normalize=True, normalize_method='minmax',
                  transformation=True, transformation_method = 'quantile', remove_multicollinearity=True, multicollinearity_threshold=0.6)

get_config("X_train")

best_model1 = compare_models()

"""#ExtraTree Regressor"""

et= create_model('et', fold=10)

et= tune_model(et,fold=10, optimize="R2")

evaluate_model(et)

interpret_model(et)

unseen_data=predict_model(et, data=data_unseen)
unseen_data.head(10)

# finalize the model**
final_best = finalize_model(et)

# save model to disk
save_model(final_best, model_name='specific-energy')

model=load_model("specific-energy")
