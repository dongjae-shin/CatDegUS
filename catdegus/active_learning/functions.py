# Following functions are used to construct transformer_X of GaussianProcess class
def scale(data, max, min):
    # Original(physical) space => Scaled space [0, 1]
    data_scaled = (data - min) / (max - min)
    return data_scaled

def descale(data, max, min):
    # Scaled space [0, 1] => Original(physical) space
    data_descaled = data * (max - min) + min
    return data_descaled

def scaler_X(X, x_range_max, x_range_min):
    scaled = X.copy()
    for i in range(X.shape[1]): # loop over columns
      scaled[:,i] = scale(scaled[:,i], x_range_max[i], x_range_min[i])
    return scaled

def descaler_X(X, x_range_max, x_range_min):
    descaled = X.copy()
    for i in range(X.shape[1]): # loop over columns
      descaled[:,i] = descale(descaled[:,i], x_range_max[i], x_range_min[i])
    return descaled