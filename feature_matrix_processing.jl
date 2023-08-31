#= combined feature matrix processing.
    Julia v1.9.1
    ScikitLearn v0.7.0
    DataFrames v1.6.1
    CSV v0.10.11
    Statistics v1.9.0
=#

using ScikitLearn
using DataFrames
using CSV
using Statistics

# scale all features to  0 - 1

# "feature_matrix_A.csv" is the combined feature matrix
df = CSV.read("feature_matrix_A.csv", DataFrame)
X = df[!, 2:end]
y = df[!, 1]

# import the MinMaxScaler from ScikitLearn.jl
@sk_import preprocessing: MinMaxScaler
scaler = MinMaxScaler()

# fit and transform the data using the scaler
scaled_data = fit_transform!(scaler, Matrix(X))
header = names(X)
X_df = DataFrame(scaled_data, header)

# remove features with less than 1% variation 
X_scaled = X_df
y_scaled = y

# function to perform feature selection based on variance
function remove_features_by_variance(X, threshold)
    variances = var(X, dims=1)
    selected_features = variances .> threshold
    return selected_features
end

# specify the variance threshold to select features
variance_threshold = 0.01
selected_features = remove_features_by_variance(Matrix(X_scaled), variance_threshold)

# convert the selected features back to a DataFrame and save to CSV "feature_matrix_F.csv"
X_selected_df = X_scaled[:, vec(selected_features)]
df_final = hcat(y_scaled, X_selected_df)
CSV.write("feature_matrix_F.csv", df_final)