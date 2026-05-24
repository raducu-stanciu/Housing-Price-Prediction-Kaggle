#%% Data Loading

import time

import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pygad
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectFromModel

#%%
# Read the data
train_data = pd.read_csv('train.csv', index_col="Id")
test_data = pd.read_csv('test.csv', index_col = 'Id')
print(f'The size of train data{train_data.shape} , the size of test data{test_data.shape}' )
#%%
print(train_data.dtypes)
#%%
train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000) 
                             & (train_data['SalePrice'] < 300_000)].index)
train_data = train_data.drop(train_data[(train_data['GarageArea'] > 1200) 
                             & (train_data['SalePrice'] < 300_000)].index)
train_data = train_data.drop(train_data[(train_data['TotalBsmtSF'] > 4000) 
                             & (train_data['SalePrice'] < 300_000)].index)
train_data = train_data.drop(train_data[(train_data['1stFlrSF'] > 4000) 
                             & (train_data['SalePrice'] < 300_000)].index)

print(f'The size of train data{train_data.shape}, the size of test data{test_data.shape}' )
# %%

#Remove rows with missing target
train_data.dropna(axis=0, subset='SalePrice', inplace=True)
y = train_data['SalePrice']
train_data.drop(['SalePrice'], axis = 1 , inplace = True)
# Break off valitation set from training set
X_train_full, X_valid_full, y_train, y_valid =  train_test_split(train_data, y,
                                                                 train_size= 0.8, 
                                                                 test_size = 0.2,
                                                                random_state=0)
low_cardinality_cols = [col for col in X_train_full.columns 
                        if X_train_full[col].nunique() < 10 and
                        (X_train_full[col].dtype == 'object' or str(X_train_full[col].dtype) == 'str')]

numeric_cols = [col for col in X_train_full.columns 
                if X_train_full[col].dtype in ['int64','float64']]
print( low_cardinality_cols)
print(numeric_cols)

#
valid_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[valid_cols].copy()
X_valid = X_valid_full[valid_cols].copy()
X_test = test_data[valid_cols].copy()
print("\n")
print(f'The size of train data{X_train.shape} , the size of test data{y.shape}' )
#%%

# Break off the columns for ordinal encoding from all categorical columns.
categorical_cols_ordinal= ['HeatingQC','PoolQC','GarageQual','GarageCond','FireplaceQu', 'KitchenQual','BsmtCond','BsmtQual','ExterQual']
categorical_cols_one_hot = [col for col in low_cardinality_cols if col  not in categorical_cols_ordinal ]
print(categorical_cols_ordinal)
print()
print(categorical_cols_one_hot)
#%%
# Define the pipeline 
one_hot_encoding = Pipeline(steps =[
    ('imputer' , SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown ='ignore', sparse_output = False))
])
ordinal_encoding = Pipeline(steps=[
    ('imputer',  SimpleImputer(strategy = 'most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
])
numerical = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy = 'mean') ),
  #  ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical , numeric_cols),
        ('cat_onehot', one_hot_encoding, categorical_cols_one_hot ), #low_cardinality_cols
        ('cat_ordinal', ordinal_encoding, categorical_cols_ordinal )
    ])
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #('model', RandomForestRegressor(n_estimators = 300, random_state=42))
    ('model', XGBRegressor(n_estimators=300, #n_estimators
                           learning_rate=0.05,  #learning_rate,
                           random_state=0))
])

#%%
def add_features(X):
    X_copy = X.copy()
    X_copy['TotalSF'] = X_copy['1stFlrSF'] + X_copy['2ndFlrSF'] + X_copy['TotalBsmtSF'] 
    X_copy['HouseAge'] = X_copy['YrSold'] - X_copy['YearBuilt']
    X_copy['TotalBath'] = X_copy['FullBath'] + (0.5 * X_copy['HalfBath']) + X_copy['BsmtFullBath']
    return X_copy
numeric_cols_extended = numeric_cols + ['TotalSF', 'HouseAge', 'TotalBath']
# %%

selector = SelectFromModel(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    threshold=0.001
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical, numeric_cols_extended), 
        ('cat_onehot', one_hot_encoding, categorical_cols_one_hot),
        ('cat_ordinal', ordinal_encoding, categorical_cols_ordinal)
    ])
my_pipeline = Pipeline(steps=[
    ('feature_eng', FunctionTransformer(add_features)),
    ('preprocessor', preprocessor),
    ('feature_selection', selector), 
    #('model', RandomForestRegressor(n_estimators=200, random_state=42)) # MAE: 17121.5369   , 16343.7798
    ('model', XGBRegressor(n_estimators=750,
                           learning_rate=0.01,
                           random_state=0,
                           tree_method = 'hist',
                           device = 'cuda'
                          )) #MAE: 17169.8359375

   
])

my_pipeline.fit(X_train, y_train)

y_pred = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_pred, y_valid)
print(f"MAE: {score}")
#%%
#  Created a pipeline only for transformations
transformation_pipeline = Pipeline(steps=my_pipeline.steps[:-1])

#  Fit & Transform, useful for GPU
X_train_final = transformation_pipeline.fit_transform(X_train, y_train).astype('float32')
X_valid_final = transformation_pipeline.transform(X_valid).astype('float32')



def fitness_func(ga_instance, solution, solution_idx):
    n_est = int(solution[0])
    lr = solution[1]
    depth = int(solution[2])

  
    final_model = XGBRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth = depth,
        random_state=0,
        tree_method='hist',
        device='cuda',
        n_jobs=-1
    )
    
    # Training
    ## X_train and X_valid for cpu
    # X_train_final and X_valid_final for GPU
    
    final_model.fit(X_train_final, y_train)
    
    # Prediction
    y_pred = final_model.predict(X_valid_final)
    score = mean_absolute_error(y_pred, y_valid)
    print(f"MAE : {score}")

    # Higher fitness for lower MAE
    fitness = 1.0 / (score + 0.01)
    return fitness

# Defining the search space (Genes)
gene_space =  [
    range(50, 2001, 50),        # n_estimators
    np.linspace(0.01, 0.2, 20),  # learning_rate
    range(2, 10)                 # max_depth
]

ga_instance = pygad.GA(
    num_generations = 20,              # Number of generations (evolution cycles)
    num_parents_mating = 5,            # Number of parents for the next generation
    fitness_func = fitness_func,
    sol_per_pop = 10,                  # Population size (configurations per generation)
    num_genes = len(gene_space),
    gene_space = gene_space, 
    parent_selection_type = 'sss',      # Steady-state selection
    keep_parents = 1,
    crossover_type = 'single_point',
    mutation_type = 'random',
    mutation_probability = 0.1
)

print("Starting genetic optimization...")
start_time = time.perf_counter()
ga_instance.run()
end_time = time.perf_counter()

# 5. Results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("-" * 30)
print(f"Best configuration found: \n n_estimators: {int(solution[0])}, learning_rate: {solution[1]:.4f}, max_depth: {int(solution[2])}")
print(f"Best MAE: {1.0/solution_fitness - 0.01:.2f}")
print(f'Optimization completed in:{((end_time-start_time)/60):.2f} min')
print("-" * 30)
##%%
# %%
#  Extract optimal parameters from the Genetic Algorithm (GA)
best_n_estimators = int(solution[0])
best_learning_rate = solution[1]
best_max_depth = int(solution[2])

#  Retrain the final model with these parameters on the transformed training data
# We use X_train_final to maintain consistency with what the GA evaluated
final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    random_state=0,
    tree_method='hist',
    device='cpu'
)

final_model.fit(X_train_final, y_train)

# The score should be identical to the best score found by the GA
preds_val = final_model.predict(X_valid_final)
final_mae = mean_absolute_error(y_valid, preds_val)
print(f"Confirmed MAE: {final_mae:.2f}")
# %%
# Created the final Pipeline
# Note: transformation_pipeline must already be .fit() on the training data
full_production_pipeline = Pipeline(steps=[
    ('transformation', transformation_pipeline),
    ('model', final_model)
])

# Save the model
joblib.dump(full_production_pipeline, 'house_price_model.pkl')
print("Model successfully saved as 'house_price_model.pkl'")
# %%
