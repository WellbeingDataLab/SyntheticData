
import pandas as pd
from synthcity.plugins import Plugins
from sdv.metadata import SingleTableMetadata

import numpy as np

from sdv.single_table import TVAESynthesizer
from sdv.single_table import CTGANSynthesizer
import imblearn

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import svm
from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import NearestNeighbors



def calc_corr_score(real_data, synth_data):
    corrdiff = 1- abs(((real_data).corr() - (synth_data.set_axis(real_data.columns,axis=1)).corr()))/2
    corrdiff = corrdiff.fillna(0)
    numpy_corrdiff = corrdiff.to_numpy()
    return np.sum(np.tril(numpy_corrdiff, k=-1),axis=(0,1))/(np.sum(np.tril(np.ones(real_data.shape[1]), k=-1),axis=(0,1))-corrdiff.isna().sum().sum())

def ks_score(real, synth):
    ks_test_results_list = []
    numerical_columns, _ = get_col_types(real)
    for col in numerical_columns:
        ks_test_result = stats.kstest(real[col].dropna(), synth[col].dropna())
        ks_test_results_list.append(ks_test_result.statistic)

    ks_score = 1- np.mean(ks_test_results_list)
    return ks_score


def ks_test(real, synth):
    ks_test_results_list = []
    numerical_columns, _ = get_col_types(real)
    for col in numerical_columns:
        ks_test_result = stats.kstest(real[col].dropna(), synth[col].dropna())
        ks_test_results_list.append(ks_test_result.statistic)
    ks_df = pd.DataFrame(np.array(ks_test_results_list)).T#, columns=numerical_columns)

    return ks_df.set_axis(numerical_columns, axis=1)


def get_col_types(df):
    sdv_metadata = SingleTableMetadata()
    sdv_metadata.detect_from_dataframe(df)
    metadata_dict = sdv_metadata.to_dict()
    numerical_columns = []
    categorical_columns = []
    for col in metadata_dict['columns']:
        if metadata_dict['columns'][col]['sdtype'] == 'numerical':
            numerical_columns.append(col)
        elif metadata_dict['columns'][col]['sdtype'] == 'categorical':
            categorical_columns.append(col)
    return numerical_columns, categorical_columns

def generate_TVAE(train_data, n_synth, com_dims=(128,64),emb_dims=32, decom_dims=(64,128),epochs=800):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    tvae_synthesizer = TVAESynthesizer(metadata,
                                               compress_dims=com_dims,
                                               decompress_dims=decom_dims,
                                               embedding_dim = emb_dims,
                                               enforce_min_max_values = False,
                                               epochs=epochs)
    tvae_synthesizer.fit(train_data)
    tvae_samples = tvae_synthesizer.sample(n_synth)

    return tvae_samples


def generate_GAN(train_data, n_synth, com_dims=(128,128),emb_dims=64, decom_dims=(128,128),epochs=800):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    gan_synthesizer = CTGANSynthesizer(metadata,
                                               enforce_min_max_values = False,
                                               epochs=epochs)
    gan_synthesizer.fit(train_data)
    tvae_samples = gan_synthesizer.sample(n_synth)

    return tvae_samples

def generate_smote(train, n, smote_category):
    n_males = np.sum(train[smote_category])
    n_females = train.shape[0] - n_males
    if n_males > n_females:
        sampling_strat = {0: int(np.ceil(n/train.shape[0])*n_males), 1: int(np.ceil(n/train.shape[0])*n_females)}
    else:
        sampling_strat = {0: int(np.ceil(n/train.shape[0])*n_females), 1: int(np.ceil(n/train.shape[0])*n_males)}


    oversample = imblearn.over_sampling.SMOTE(sampling_strategy=sampling_strat,random_state=19)
    X, y= oversample.fit_resample(train.drop(smote_category,axis=1), train[smote_category])
    X.insert(1,'Sex',y)
    smote_samples = X.iloc[train.shape[0]:,:]

    oversample = imblearn.over_sampling.ADASYN(sampling_strategy=sampling_strat,random_state=19)
    X, y= oversample.fit_resample(train.drop(smote_category,axis=1), train[smote_category])
    X.insert(1,'Sex',y)
    adasyn_samples = X.iloc[train.shape[0]:,:]

    oversample = imblearn.over_sampling.SVMSMOTE(sampling_strategy=sampling_strat,random_state=19)
    X, y= oversample.fit_resample(train.drop(smote_category,axis=1), train[smote_category])
    X.insert(1,'Sex',y)
    svmsmote_samples = X.iloc[train.shape[0]:,:]

    _, categorical_columns = get_col_types(train)

    for col in categorical_columns:
        smote_samples.loc[:, (col)] = smote_samples[col].round(0).copy()
        adasyn_samples.loc[:, (col)] = adasyn_samples[col].round(0).copy()
        svmsmote_samples.loc[:, (col)] = svmsmote_samples[col].round(0).copy()
    
    return smote_samples, adasyn_samples, svmsmote_samples


def generate_synthetic_data(train_set, n_samples=1000):
    tvae_samples = generate_TVAE(train_set, n_samples)
    gan_samples = generate_GAN(train_set,n_samples)
    print("tvae ctgan ready")

    syncity_adsgan = Plugins().get("adsgan")
    syncity_adsgan.fit(train_set)
    adsgan_samples = syncity_adsgan.generate(n_samples)
    X, y = adsgan_samples.unpack()
    adsgan_samples = pd.concat([X,y],axis=1)
    print("adsgan ready")

    bayes_net = Plugins().get("bayesian_network")
    bayes_net.fit(train_set, encoder_noise_scale = 0)
    bayes_net_samples = bayes_net.generate(n_samples)
    X, y = bayes_net_samples.unpack()
    bayes_net_samples = pd.concat([X,y],axis=1)
    print("bayes ready")

    arfgan = Plugins().get("arf")
    arfgan.fit(train_set)
    arfgan_samples = arfgan.generate(n_samples)
    X, y = arfgan_samples.unpack()
    arfgan_samples = pd.concat([X,y],axis=1)
    print("arfgan ready")

    norm_flows = Plugins().get("nflow", n_iter = 1000)
    norm_flows.fit(train_set)
    norm_flows_samples = norm_flows.generate(n_samples)
    X, y = norm_flows_samples.unpack()
    norm_flows_samples = pd.concat([X,y],axis=1)
    print("nflows ready")

    rtvae = Plugins().get("rtvae", n_iter = 1000)
    rtvae.fit(train_set)
    rtvae_samples = rtvae.generate(n_samples)
    X, y = rtvae_samples.unpack()
    rtvae_samples = pd.concat([X,y],axis=1)
    print("rtvae")

    return tvae_samples, gan_samples, adsgan_samples, bayes_net_samples, arfgan_samples, norm_flows_samples, rtvae_samples


def visualize_and_test(real_data, synthetic_data, variable_1, variable_2):

    ks = ks_score(real_data, synthetic_data)
    corr = calc_corr_score(real_data, synthetic_data)
    plt.scatter(synthetic_data[variable_1], synthetic_data[variable_2], label = 'synthetic')
    plt.scatter(real_data[variable_1],real_data[variable_2], label = 'real')
    plt.xlabel(variable_1)
    plt.ylabel(variable_2)

    plt.legend()

    return ks, corr


def scale_numerical(samples, train_set, test_set):
    numerical_columns, categorical_columns = get_col_types(train_set)

    samples_num = samples[numerical_columns]
    samples_cat = samples[categorical_columns]

    train_num = train_set[numerical_columns]
    train_cat = train_set[categorical_columns]

    test_num = test_set[numerical_columns]
    test_cat = test_set[categorical_columns]

    real_and_fake_df = pd.concat([train_num, samples_num], axis=0)

    scaler_real_fake = StandardScaler()
    scaler_real_fake.fit(real_and_fake_df)

    train_continuous_scaled = scaler_real_fake.transform(train_num)
    samples_continuous_scaled = scaler_real_fake.transform(samples_num)
    test_continuous_scaled = scaler_real_fake.transform(test_num)

    samples_scaled = pd.concat([samples_cat.reset_index(drop=True), pd.DataFrame(samples_continuous_scaled, columns=numerical_columns)], axis=1)
    train_scaled = pd.concat([train_cat.reset_index(drop=True), pd.DataFrame(train_continuous_scaled, columns=numerical_columns)], axis=1)
    test_scaled = pd.concat([test_cat.reset_index(drop=True), pd.DataFrame(test_continuous_scaled, columns=numerical_columns)], axis=1)

    return samples_scaled, train_scaled, test_scaled, scaler_real_fake


def utility_test(synth_data, real_data, test_data):
    synth, train_real, test_real, scaler = scale_numerical(synth_data, real_data, test_data)
    numerical_columns, categorical_columns = get_col_types(real_data)

    mse_list_trtr_GB = []
    mse_list_tstr_GB = []

    mse_list_trtr_RF = []
    mse_list_tstr_RF = []

    mse_list_trtr_SVM = []
    mse_list_tstr_SVM = []

    mse_list_trtr_NN = []
    mse_list_tstr_NN = []

    mse_list_trtr_DT = []
    mse_list_tstr_DT = []

    mse_list_trtr_MLP = []
    mse_list_tstr_MLP = []

    mse_list_trtr_reg = []
    mse_list_tstr_reg = []

    r2_list_trtr_GB = []
    r2_list_tstr_GB = []

    r2_list_trtr_RF = []
    r2_list_tstr_RF = []

    r2_list_trtr_SVM = []
    r2_list_tstr_SVM = []

    r2_list_trtr_NN = []
    r2_list_tstr_NN = []

    r2_list_trtr_DT = []
    r2_list_tstr_DT = []

    r2_list_trtr_MLP = []
    r2_list_tstr_MLP = []

    r2_list_trtr_reg = []
    r2_list_tstr_reg = []

    f1_list_trtr_GB = []
    f1_list_tstr_GB = []

    f1_list_trtr_RF = []
    f1_list_tstr_RF = []

    f1_list_trtr_SVM = []
    f1_list_tstr_SVM = []

    f1_list_trtr_NN = []
    f1_list_tstr_NN = []

    f1_list_trtr_DT = []
    f1_list_tstr_DT = []

    f1_list_trtr_MLP = []
    f1_list_tstr_MLP = []

    f1_list_trtr_reg = []
    f1_list_tstr_reg = []

    for y_column in real_data.columns:
        y_test = test_real[y_column]
        X_test = test_real.copy().drop(y_column, axis=1)
       
        y_train_real = train_real[y_column]
        X_train_real = train_real.copy().drop(y_column, axis=1)
        y_train_synth = synth[y_column]
        X_train_synth = synth.copy().drop(y_column, axis=1)

        if y_column in categorical_columns:
            grad_boost_synth = GradientBoostingClassifier()
            grad_boost_synth.fit(X_train_synth, y_train_synth)
            grad_boost_real = GradientBoostingClassifier()
            grad_boost_real.fit(X_train_real, y_train_real)

            rf_synth = RandomForestClassifier()
            rf_synth.fit(X_train_synth, y_train_synth)
            rf_real = RandomForestClassifier()
            rf_real.fit(X_train_real, y_train_real)

            support_vm_synth = svm.SVC()
            support_vm_synth.fit(X_train_synth, y_train_synth)
            support_vm_real = svm.SVC()
            support_vm_real.fit(X_train_real, y_train_real)

            neigh_synth = KNeighborsClassifier(n_neighbors=3)
            neigh_synth.fit(X_train_synth, y_train_synth)
            neigh_real = KNeighborsClassifier(n_neighbors=3)
            neigh_real.fit(X_train_real, y_train_real)

            decision_tree_synth = tree.DecisionTreeClassifier()
            decision_tree_synth.fit(X_train_synth, y_train_synth)
            decision_tree_real = tree.DecisionTreeClassifier()
            decision_tree_real.fit(X_train_real, y_train_real)

            mlp_synth = MLPClassifier(max_iter=500)
            mlp_synth.fit(X_train_synth, y_train_synth)
            mlp_real = MLPClassifier(max_iter=500)
            mlp_real.fit(X_train_real, y_train_real)

            reg_synth = LogisticRegression()
            reg_synth.fit(X_train_synth, y_train_synth)
            reg_real = LogisticRegression()
            reg_real.fit(X_train_real, y_train_real)

            f1_s_GB = f1_score(y_test, grad_boost_synth.predict(X_test),average='micro')
            f1_r_GB = f1_score(y_test, grad_boost_real.predict(X_test),average='micro')

            f1_s_RF = f1_score(y_test, rf_synth.predict(X_test),average='micro')
            f1_r_RF = f1_score(y_test, rf_real.predict(X_test),average='micro')

            f1_s_SVM = f1_score(y_test, support_vm_synth.predict(X_test),average='micro')
            f1_r_SVM = f1_score(y_test, support_vm_real.predict(X_test),average='micro')

            f1_s_NN = f1_score(y_test, neigh_synth.predict(X_test),average='micro')
            f1_r_NN = f1_score(y_test, neigh_real.predict(X_test),average='micro')

            f1_s_DT = f1_score(y_test, decision_tree_synth.predict(X_test),average='micro')
            f1_r_DT = f1_score(y_test, decision_tree_real.predict(X_test),average='micro')

            f1_s_MLP = f1_score(y_test, mlp_synth.predict(X_test),average='micro')
            f1_r_MLP = f1_score(y_test, mlp_real.predict(X_test),average='micro')

            f1_s_REG = f1_score(y_test, reg_synth.predict(X_test),average='micro')
            f1_r_REG = f1_score(y_test, reg_real.predict(X_test),average='micro')

            f1_list_trtr_GB.append(f1_r_GB)
            f1_list_tstr_GB.append(f1_s_GB)
            f1_list_trtr_RF.append(f1_r_RF)
            f1_list_tstr_RF.append(f1_s_RF)
            f1_list_trtr_SVM.append(f1_r_SVM)
            f1_list_tstr_SVM.append(f1_s_SVM)
            f1_list_trtr_NN.append(f1_r_NN)
            f1_list_tstr_NN.append(f1_s_NN)
            f1_list_trtr_DT.append(f1_r_DT)
            f1_list_tstr_DT.append(f1_s_DT)
            f1_list_trtr_MLP.append(f1_r_MLP)
            f1_list_tstr_MLP.append(f1_s_MLP)
            f1_list_trtr_reg.append(f1_r_REG)
            f1_list_tstr_reg.append(f1_s_REG)


        else:
            grad_boost_synth = GradientBoostingRegressor()
            grad_boost_synth.fit(X_train_synth, y_train_synth)
            grad_boost_real = GradientBoostingRegressor()
            grad_boost_real.fit(X_train_real, y_train_real)

            rf_synth = RandomForestRegressor()
            rf_synth.fit(X_train_synth, y_train_synth)
            rf_real = RandomForestRegressor()
            rf_real.fit(X_train_real, y_train_real)

            support_vm_synth = svm.SVR()
            support_vm_synth.fit(X_train_synth, y_train_synth)
            support_vm_real = svm.SVR()
            support_vm_real.fit(X_train_real, y_train_real)

            neigh_synth = KNeighborsRegressor(n_neighbors=3)
            neigh_synth.fit(X_train_synth, y_train_synth)
            neigh_real = KNeighborsRegressor(n_neighbors=3)
            neigh_real.fit(X_train_real, y_train_real)

            decision_tree_synth = tree.DecisionTreeRegressor()
            decision_tree_synth.fit(X_train_synth, y_train_synth)
            decision_tree_real = tree.DecisionTreeRegressor()
            decision_tree_real.fit(X_train_real, y_train_real)

            mlp_synth = MLPRegressor(max_iter=500)
            mlp_synth.fit(X_train_synth, y_train_synth)
            mlp_real = MLPRegressor(max_iter=500)
            mlp_real.fit(X_train_real, y_train_real)

            reg_synth = LinearRegression()
            reg_synth.fit(X_train_synth, y_train_synth)
            reg_real = LinearRegression()
            reg_real.fit(X_train_real, y_train_real)


            mse_s_GB = mean_squared_error(y_test, grad_boost_synth.predict(X_test))
            mse_r_GB = mean_squared_error(y_test, grad_boost_real.predict(X_test))
            r2_s_GB = r2_score(y_test, grad_boost_synth.predict(X_test))
            r2_r_GB = r2_score(y_test, grad_boost_real.predict(X_test))
            
            
            mse_s_RF = mean_squared_error(y_test, rf_synth.predict(X_test))
            mse_r_RF = mean_squared_error(y_test, rf_real.predict(X_test))
            r2_s_RF = r2_score(y_test, rf_synth.predict(X_test))
            r2_r_RF = r2_score(y_test, rf_real.predict(X_test))

            mse_s_SVM = mean_squared_error(y_test, support_vm_synth.predict(X_test))
            mse_r_SVM = mean_squared_error(y_test, support_vm_real.predict(X_test))
            r2_s_SVM = r2_score(y_test, support_vm_synth.predict(X_test))
            r2_r_SVM = r2_score(y_test, support_vm_real.predict(X_test))

            mse_s_NN = mean_squared_error(y_test, neigh_synth.predict(X_test))
            mse_r_NN = mean_squared_error(y_test, neigh_real.predict(X_test))
            r2_s_NN = r2_score(y_test, neigh_synth.predict(X_test))
            r2_r_NN = r2_score(y_test, neigh_real.predict(X_test))

            mse_s_DT = mean_squared_error(y_test, decision_tree_synth.predict(X_test))
            mse_r_DT = mean_squared_error(y_test, decision_tree_real.predict(X_test))
            r2_s_DT = r2_score(y_test, decision_tree_synth.predict(X_test))
            r2_r_DT = r2_score(y_test, decision_tree_real.predict(X_test))

            mse_s_MLP = mean_squared_error(y_test, mlp_synth.predict(X_test))
            mse_r_MLP = mean_squared_error(y_test, mlp_real.predict(X_test))
            r2_s_MLP = r2_score(y_test, mlp_synth.predict(X_test))
            r2_r_MLP = r2_score(y_test, mlp_real.predict(X_test))

            mse_s_REG = mean_squared_error(y_test, reg_synth.predict(X_test))
            mse_r_REG = mean_squared_error(y_test, reg_real.predict(X_test))
            r2_s_REG = r2_score(y_test, reg_synth.predict(X_test))
            r2_r_REG = r2_score(y_test, reg_real.predict(X_test))

            mse_list_trtr_GB.append(mse_r_GB)
            mse_list_tstr_GB.append(mse_s_GB)
            r2_list_trtr_GB.append(r2_r_GB)
            r2_list_tstr_GB.append(r2_s_GB)

            mse_list_trtr_RF.append(mse_r_RF)
            mse_list_tstr_RF.append(mse_s_RF)
            r2_list_trtr_RF.append(r2_r_RF)
            r2_list_tstr_RF.append(r2_s_RF)

            mse_list_trtr_SVM.append(mse_r_SVM)
            mse_list_tstr_SVM.append(mse_s_SVM)
            r2_list_trtr_SVM.append(r2_r_SVM)
            r2_list_tstr_SVM.append(r2_s_SVM)

            mse_list_trtr_NN.append(mse_r_NN)
            mse_list_tstr_NN.append(mse_s_NN)
            r2_list_trtr_NN.append(r2_r_NN)
            r2_list_tstr_NN.append(r2_s_NN)

            mse_list_trtr_DT.append(mse_r_DT)
            mse_list_tstr_DT.append(mse_s_DT)
            r2_list_trtr_DT.append(r2_r_DT)
            r2_list_tstr_DT.append(r2_s_DT)

            mse_list_trtr_MLP.append(mse_r_MLP)
            mse_list_tstr_MLP.append(mse_s_MLP)
            r2_list_trtr_MLP.append(r2_r_MLP)
            r2_list_tstr_MLP.append(r2_s_MLP)

            mse_list_trtr_reg.append(mse_r_REG)
            mse_list_tstr_reg.append(mse_s_REG)
            r2_list_trtr_reg.append(r2_r_REG)
            r2_list_tstr_reg.append(r2_s_REG)

    mse_trtr = np.array([mse_list_trtr_GB, mse_list_trtr_RF, mse_list_trtr_SVM, mse_list_trtr_NN,mse_list_trtr_DT, mse_list_trtr_MLP, mse_list_trtr_reg])
    mse_tstr = np.array([mse_list_tstr_GB, mse_list_tstr_RF, mse_list_tstr_SVM, mse_list_tstr_NN,mse_list_tstr_DT, mse_list_tstr_MLP, mse_list_tstr_reg])

    r2_trtr = np.array([r2_list_trtr_GB, r2_list_trtr_RF, r2_list_trtr_SVM, r2_list_trtr_NN,r2_list_trtr_DT, r2_list_trtr_MLP, r2_list_trtr_reg])
    r2_tstr = np.array([r2_list_tstr_GB, r2_list_tstr_RF, r2_list_tstr_SVM, r2_list_tstr_NN,r2_list_tstr_DT, r2_list_tstr_MLP, r2_list_tstr_reg])

    f1_trtr = np.array([f1_list_trtr_GB, f1_list_trtr_RF, f1_list_trtr_SVM, f1_list_trtr_NN,f1_list_trtr_DT, f1_list_trtr_MLP, f1_list_trtr_reg])
    f1_tstr = np.array([f1_list_tstr_GB, f1_list_tstr_RF, f1_list_tstr_SVM, f1_list_tstr_NN,f1_list_tstr_DT, f1_list_tstr_MLP, f1_list_tstr_reg])

    mse_trtr = pd.DataFrame(mse_trtr, columns=numerical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    mse_tstr = pd.DataFrame(mse_tstr, columns=numerical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    r2_trtr = pd.DataFrame(r2_trtr, columns=numerical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    r2_tstr = pd.DataFrame(r2_tstr, columns=numerical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    f1_trtr = pd.DataFrame(f1_trtr, columns=categorical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    f1_tstr = pd.DataFrame(f1_tstr, columns=categorical_columns, index=['GradBoost', 'RandForest', 'SVM', 'NearNeigh', 'DecTree', 'MLP', 'Regress'])
    numerical_metric_df = pd.concat([mse_trtr, mse_tstr, r2_trtr, r2_tstr], keys=["MSE_trtr","MSE_tstr", "R2_trtr", "R2_tstr"])
    categorical_metric_df = pd.concat([f1_trtr, f1_tstr], keys=["F1_trtr", "F1_tstr"])


    return numerical_metric_df, categorical_metric_df
        


def distance_score(train, test, synth):
    nbrs_train = NearestNeighbors(n_neighbors=2).fit(train)
    nbrs_test = NearestNeighbors(n_neighbors=2).fit(test)

    nearest_distance_train, _ = nbrs_train.kneighbors(synth, n_neighbors=1)
    nearest_distance_test, _ = nbrs_test.kneighbors(synth, n_neighbors=1)
    
    mean_nearest_train = np.mean(nearest_distance_train)
    mean_nearest_test = np.mean(nearest_distance_test)

    return mean_nearest_train, mean_nearest_test

def naive_attack(train, test, synth):
    clf =  GradientBoostingClassifier()
    # train = train_df.insert(0,'y',1)
    # test = test_df.insert(0,'y',0)
    train.insert(0,'y',1)
    test.insert(0,'y',0)
    full_df = pd.concat([train, test])
    X_train = full_df.drop('y',axis=1)
    y_train = full_df['y']
    

    clf.fit(X_train, y_train)
    return np.sum(clf.predict(synth))/synth.shape[0]


def nearest_neighbor_distance_ratio(real_df_train_list, real_df_test_list, synthetic_df_list):
    test_mean_2close_ratio_list = []
    synth_mean_2close_ratio_list = []
    test_std_2close_ratio_list = []
    synth_std_2close_ratio_list = []

    test_max_2close_ratio_list = []
    synth_max_2close_ratio_list = []
    test_min_2close_ratio_list = []
    synth_min_2close_ratio_list = []

    ratios_test = []
    ratios_synth = []

    for i in range(0,len(real_df_train_list)):
        scaler_minmax = MinMaxScaler()
        if 'ID' in real_df_train_list[i].columns:
            real_df_train_list[i] = real_df_train_list[i].drop('ID', axis=1)
            real_df_test_list[i] =real_df_test_list[i].drop('ID', axis=1)
        real_df_train_scaled = scaler_minmax.fit_transform(real_df_train_list[i])
        real_df_test_scaled = scaler_minmax.transform(real_df_test_list[i])
        synth_scaled = scaler_minmax.transform(synthetic_df_list[i])

        two_closest_ratio_list_real = []
        for test_dude in real_df_test_scaled:
            test_dude_distances = []
            for real_dude in real_df_train_scaled:
                test_dude_distances.append(np.linalg.norm(real_dude-test_dude))
            test_dude_distances_sorted = np.sort(np.array(test_dude_distances))
            two_closest_ratio_list_real.append(test_dude_distances_sorted[0]/test_dude_distances_sorted[1])

        test_mean_2close_ratio_list.append(np.mean(two_closest_ratio_list_real))
        test_std_2close_ratio_list.append(np.std(two_closest_ratio_list_real))
        test_max_2close_ratio_list.append(np.max(two_closest_ratio_list_real))
        test_min_2close_ratio_list.append(np.min(two_closest_ratio_list_real))

        ratios_test.append(two_closest_ratio_list_real)

        two_closest_ratio_list_synth = []
        for synth_dude in synth_scaled:
            synth_dude_distances = []
            for real_dude in real_df_train_scaled:
                synth_dude_distances.append(np.linalg.norm(real_dude-synth_dude))
            synth_dude_distances_sorted = np.sort(np.array(synth_dude_distances))
            two_closest_ratio_list_synth.append(synth_dude_distances_sorted[0]/synth_dude_distances_sorted[1])

        synth_mean_2close_ratio_list.append(np.mean(two_closest_ratio_list_synth))
        synth_std_2close_ratio_list.append(np.std(two_closest_ratio_list_synth))
        synth_max_2close_ratio_list.append(np.max(two_closest_ratio_list_synth))
        synth_min_2close_ratio_list.append(np.min(two_closest_ratio_list_synth))

        ratios_synth.append(two_closest_ratio_list_synth)

    return ratios_synth, ratios_test
    

def membership_attack(synth_data, real_data, test_data):
    synth, train_real, test_real, scaler = scale_numerical(synth_data, real_data, test_data)
    # numerical_columns, categorical_columns = utils.get_col_types(real_data)
    cols = synth_data.columns

    rmse_list_s_REG = []
    rmse_list_r_REG = []
    rmse_list_s_GB = []
    rmse_list_r_GB = []

    mae_list_s_REG = []
    mae_list_r_REG = []
    mae_list_s_GB = []
    mae_list_r_GB = []

    rmse_list_test_test_GB = []
    rmse_list_train_test_GB = []
    rmse_list_test_test_REG = []
    rmse_list_train_test_REG = []

    grad_boost_synth_predicts_train_test_list = []
    grad_boost_synth_predicts_test_test_list =[]
    grad_boost_synth_true_train_test_list = []
    grad_boost_synth_true_test_test_list =[]

    for i, y_column in enumerate(cols):

        y_train_real = train_real[y_column]
        X_train_real = train_real.copy().drop(y_column, axis=1)
        y_train_synth = synth[y_column]
        X_train_synth = synth.copy().drop(y_column, axis=1)

        # else:
        grad_boost_synth = GradientBoostingRegressor()
        grad_boost_synth.fit(X_train_synth, y_train_synth)

        reg_synth = LinearRegression()
        reg_synth.fit(X_train_synth, y_train_synth)
        train_test_i = np.random.randint(0, train_real.shape[0], 30)
        test_test_i = np.random.randint(0, test_real.shape[0], 30)
        train_test = train_real.iloc[train_test_i]
        test_test = test_real.iloc[test_test_i]

        grad_boost_synth_predicts_real = grad_boost_synth.predict(train_real.drop(y_column, axis=1).drop(train_test_i))
        grad_boost_synth_predicts_test = grad_boost_synth.predict(test_real.drop(y_column, axis=1).drop(test_test_i))

        grad_boost_synth_predicts_train_test = grad_boost_synth.predict(train_test.drop(y_column, axis=1))
        grad_boost_synth_predicts_test_test = grad_boost_synth.predict(test_test.drop(y_column, axis=1))

        grad_boost_synth_predicts_train_test_list.append(grad_boost_synth_predicts_train_test)
        grad_boost_synth_predicts_test_test_list.append(grad_boost_synth_predicts_test_test)

        grad_boost_synth_true_train_test_list.append(train_real.iloc[train_test_i][y_column])
        grad_boost_synth_true_test_test_list.append(test_real.iloc[test_test_i][y_column])

        rmse_s_GB = root_mean_squared_error(train_real[y_column].drop(train_test_i), grad_boost_synth_predicts_real)
        rmse_r_GB = root_mean_squared_error(test_real[y_column].drop(test_test_i), grad_boost_synth_predicts_test)

        mae_s_GB = mean_absolute_error(train_real[y_column].drop(train_test_i), grad_boost_synth_predicts_real)
        mae_r_GB = mean_absolute_error(test_real[y_column].drop(test_test_i), grad_boost_synth_predicts_test)

        train_test_rsme_gb =root_mean_squared_error(train_test[y_column], grad_boost_synth_predicts_train_test)
        test_test_rsme_gb =root_mean_squared_error(test_test[y_column], grad_boost_synth_predicts_test_test)


        reg_synth_predicts_real = reg_synth.predict(train_real.drop(y_column, axis=1).drop(train_test_i))
        reg_synth_predicts_test = reg_synth.predict(test_real.drop(y_column, axis=1).drop(test_test_i))

        reg_synth_predicts_train_test = reg_synth.predict(train_test.drop(y_column, axis=1))
        reg_synth_predicts_test_test = reg_synth.predict(test_test.drop(y_column, axis=1))

        rmse_s_reg = root_mean_squared_error(train_real[y_column].drop(train_test_i), reg_synth_predicts_real)
        rmse_r_reg = root_mean_squared_error(test_real[y_column].drop(test_test_i), reg_synth_predicts_test)

        mae_s_reg = mean_absolute_error(train_real[y_column].drop(train_test_i), reg_synth_predicts_real)
        mae_r_reg = mean_absolute_error(test_real[y_column].drop(test_test_i), reg_synth_predicts_test)

        train_test_rsme_reg =root_mean_squared_error(train_test[y_column], reg_synth_predicts_train_test)
        test_test_rsme_reg =root_mean_squared_error(test_test[y_column], reg_synth_predicts_test_test)

        rmse_list_s_REG.append(rmse_s_reg)
        rmse_list_r_REG.append(rmse_r_reg)
        rmse_list_r_GB.append(rmse_r_GB)
        rmse_list_s_GB.append(rmse_s_GB)
        
        mae_list_s_REG.append(mae_s_reg)
        mae_list_r_REG.append(mae_r_reg)
        mae_list_r_GB.append(mae_r_GB)
        mae_list_s_GB.append(mae_s_GB)

        rmse_list_train_test_GB.append(train_test_rsme_gb)
        rmse_list_test_test_GB.append(test_test_rsme_gb)

        rmse_list_train_test_REG.append(train_test_rsme_reg)
        rmse_list_test_test_REG.append(test_test_rsme_reg)

    mse_trtr = np.array([rmse_list_r_GB, rmse_list_r_REG])
    mse_tstr = np.array([rmse_list_s_GB, rmse_list_s_REG])

    mae_trtr = np.array([mae_list_r_GB, mae_list_r_REG])
    mae_tstr = np.array([mae_list_s_GB, mae_list_s_REG])

    rmse_train_test = np.array([rmse_list_train_test_GB,rmse_list_train_test_REG])
    rmse_test_test = np.array([rmse_list_test_test_GB,rmse_list_test_test_REG])

    mse_trtr = pd.DataFrame(mse_trtr, columns=cols, index=['GradBoost', 'Regress'])
    mse_tstr = pd.DataFrame(mse_tstr, columns=cols, index=['GradBoost', 'Regress'])

    rmse_train_test = pd.DataFrame(rmse_train_test, columns=cols, index=['GradBoost', 'Regress'])
    rmse_test_test = pd.DataFrame(rmse_test_test, columns=cols, index=['GradBoost', 'Regress'])

    mae_trtr = pd.DataFrame(mae_trtr, columns=cols, index=['GradBoost', 'Regress'])
    mae_tstr = pd.DataFrame(mae_tstr, columns=cols, index=['GradBoost', 'Regress'])

    grad_boost_synth_predicts_train_test_list

    attacks = pd.concat([rmse_train_test, rmse_test_test], keys=["rmse_train_test","rmse_test_test"])
    numerical_metric_df = pd.concat([mse_trtr, mse_tstr], keys=["rmsetrtr","rmsetstr"])


    return grad_boost_synth_predicts_train_test_list, grad_boost_synth_predicts_test_test_list ,grad_boost_synth_true_train_test_list, grad_boost_synth_true_test_test_list