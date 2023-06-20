import pandas as pd


def scaled_datasets(train_df, valid_df, scaler, continuous_feat):
    if continuous_feat == None:
        return train_df, valid_df

    else:
        scaled_train, scaled_valid = scaling(train_df, valid_df, scaler, continuous_feat)  # scaler = StandardScaler()

        scaled_train_df = pd.concat([train_df, scaled_train], axis=1)
        scaled_train_df = scaled_train_df.drop(continuous_feat, axis=1)

        scaled_valid_df = pd.concat([valid_df, scaled_valid], axis=1)
        scaled_valid_df = scaled_valid_df.drop(continuous_feat, axis=1)

        return scaled_train_df, scaled_valid_df


def scaling(train, valid, scaler, continuous_feat):
    contin_train = train[continuous_feat]
    contin_valid = valid[continuous_feat]

    fitted_scaler = scaler.fit(contin_train)

    scaled_t = fitted_scaler.fit_transform(contin_train)  # scaled_t : numpy.ndarray
    scaled_t = pd.DataFrame(scaled_t, columns=['scaled_' + feat for feat in continuous_feat])

    scaled_v = fitted_scaler.fit_transform(contin_valid)
    scaled_v = pd.DataFrame(scaled_v, columns=['scaled_' + feat for feat in continuous_feat])

    scaled_train = get_scaled_df(scaled_t)
    scaled_valid = get_scaled_df(scaled_v)

    return scaled_train, scaled_valid


def get_scaled_df(scaled_df):
    for i in range(len(scaled_df.columns)):
        values = scaled_df[scaled_df.columns[i]]
        for j in range(len(values)):
            if str(values[j]) == 'nan':
                values[j] = 0.0
    return scaled_df