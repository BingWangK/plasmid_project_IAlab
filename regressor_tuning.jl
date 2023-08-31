#= base models hyperparameters tuning
	Julia v1.9.1
	MLJ v0.19.2
	DataFrames v1.6.1
       CSV v0.10.11
	Plots v1.38.17
       MLJEnsembles v0.3.3
	Random
       Resample v1.0.2
=#

using MLJ
using DataFrames
using CSV
using Plots
using MLJEnsembles
using Random
using Resample

train_id="run_id"

# split dataset for tuning. "feature_matrix_F.csv" is the data matrix containing target Psum and the features
df_origin = CSV.read("feature_matrix_F.csv", DataFrame)
train, df_test = partition(df_origin, 0.8, shuffle=true, rng=123)
sort!(df_test, :Psum)
df_minor = filter(row -> row.Psum > 1.2, train)
df_minor2 = filter(row -> row.Psum > 2.5, train)
num_minor = nrows(df_minor)
num_minor2 = nrows(df_minor2)
newdata = smote(df_minor, floor(Int64, round(num_minor*2)))
newdata2 = smote(df_minor2, floor(Int64, round(num_minor2*1)))
df_train = vcat(train, DataFrame(newdata), DataFrame(newdata2)
) # the new train dataset supplemented with SMOTE data
X = df_train[!, 2:end]
y = df_train[!, 1]
X_test = df_test[!, 2:end]
y_test = df_test[!, 1]

# load model that needs to be tuned. We tune the models one by one.

#cat = @MLJ.load CatBoostRegressor pkg=CatBoost
#model=cat(loss_function="Quantile", iterations=1500, depth=6, learning_rate=0.1, l2_leaf_reg=1, border_count=10)

#random = @MLJ.load RandomForestRegressor pkg=DecisionTree
#model=random(n_trees = 130, n_subfeatures = 70, max_depth = 31, min_samples_split =2)

#decision = @MLJ.load DecisionTreeRegressor pkg=DecisionTree
#model=decision(max_depth = 23, min_samples_split = 4, n_subfeatures = 70, min_samples_leaf = 10)

#evo = @MLJ.load EvoTreeRegressor pkg=EvoTrees #v0.14.11
#model=evo(loss=:tweedie, nrounds = 70, gamma = 5.1, eta = 0.5, max_depth =7, colsample = 0.822, nbins = 18)

#boost = @MLJ.load XGBoostRegressor pkg=XGBoost
#model=boost(eta = 0.2556, lambda=10, gamma=2, max_depth =5, colsample_bylevel = 0.6)

#light = @MLJ.load LGBMRegressor pkg=LightGBM
#model = light(num_iterations = 100, max_depth = 5, feature_fraction = 0.8, learning_rate = 0.055,min_data_in_leaf = 25, lambda_l1 = 1, min_gain_to_split = 1, time_out = 14400)

# The ranges of hyperparameters for tuning

# RandomForestRegressor
#n_trees_range = range(model, :n_trees, lower=50, upper=200, unit=50)
#max_depth_range = range(model, :max_depth, lower=15, upper=40, unit=5)
#min_samples_split_range = range(model, :min_samples_split, lower=2, upper=6, unit=2)
#n_subfeatures_range = range(model, :n_subfeatures, lower=15, upper=114, unit=10)
#min_samples_leaf_range = range(model, :min_samples_leaf, lower=1, upper=5, unit=2)
#min_purity_increase_range = range(model, :min_purity_increase, lower=0, upper=0.1, unit=0.01)

# DecisionTreeRegressor
#max_depth_range = range(model, :max_depth, lower=10, upper=40)
#min_samples_split_range = range(model, :min_samples_split, lower=2, upper=5)
#n_subfeatures_range = range(model, :n_subfeatures, lower=20, upper=100)
#min_samples_leaf_range = range(model, :min_samples_leaf, lower=5, upper=10)
#merge_purity_threshold_range = range(model, :merge_purity_threshold, lower=0.5, upper=1) # need to set post_prune = true

# EvoTreeRegressor
#nrounds_range = range(model, :nrounds, lower=50, upper=200, unit=50)
#gamma_range = range(model, :gamma, lower=0.2, upper=0.6, unit=0.1)
#eta_range = range(model, :eta, lower=0.2, upper=0.6, unit=0.1)
#max_depth_range = range(model, :max_depth, lower=2, upper=6, unit=1)
#colsample_range = range(model, :colsample, lower=0.5, upper=1, unit=0.1)
#nbins_range = range(model, :nbins, lower=10, upper=35, unit=5)

# CatBoostRegressor
#iterations_range = range(model, :iterations, lower=500, upper=1500, unit=100)
#depth_range = range(model, :depth, lower=3, upper=9, unit=1)
#learning_rate_range = range(model, :learning_rate, lower=0.01, upper=0.1, unit=0.01)
#l2_leaf_reg_range = range(model, :l2_leaf_reg, lower=0.1, upper=1, unit=0.1)

# XGBoostRegressor
#gamma_range = range(model, :gamma, lower=2, upper=7, unit=0.5)
#eta_range = range(model, :eta, lower=0.1, upper=0.6, unit=0.1)
#max_depth_range = range(model, :max_depth, lower=5, upper=11, unit=1)
#colsample_bylevel_range = range(model, :colsample_bylevel, lower=0.6, upper=1, unit=0.1)

# LGBMRegressor
#num_iterations_range = range(model, :num_iterations, lower=50, upper=150, unit=20)
#max_depth_range = range(model, :max_depth, lower=10, upper=40, unit=5)
#feature_fraction_range = range(model, :feature_fraction, lower=0.5, upper=1, unit=0.1)
#learning_rate_range = range(model, :learning_rate, lower=0.1, upper=1, unit=0.1)
#min_data_in_leaf_range = range(model, :min_data_in_leaf, lower=10, upper=30, unit=5)
#lambda_l1_range = range(model, :lambda_l1, lower=0, upper=5, unit=1)

tuning_model = TunedModel(model=model,
							tuning=Grid(),
							resampling=CV(nfolds=5, shuffle=true),
							range=[learning_rate_range], # update the range to desired hyperparameters 
							measure=rms
							)

# create a machine with the tuned model
mach = machine(tuning_model, X, y)
MLJ.fit!(mach)

# plot the optimizations
optimization_fig = plot(mach)
savefig(optimization_fig, "optimization_fig_$train_id.pdf")

# retrieve the best model with optimal hyperparameters
best_model_with_hyperparams = fitted_params(mach).best_model
# save the optimized hyperparameters. The hyperparameters need to be updated according to the model
file = open("optimal_hyperparameters_$train_id.txt", "w")
write(file, "Hyperparameters tuning:\n")
write(file, "iterations = $(best_model_with_hyperparams.iterations)\n")
write(file, "depth = $(best_model_with_hyperparams.depth)\n")
write(file, "learning_rate = $(best_model_with_hyperparams.learning_rate)\n")
write(file, "l2_leaf_reg = $(best_model_with_hyperparams.l2_leaf_reg)\n")
# Close the file
close(file)

# test the model on the testing dataset
final_model = machine(best_model_with_hyperparams, X, y)
MLJ.fit!(final_model)

y_pred = MLJ.predict(final_model, X_test)
mse = mean((y_pred .- y_test).^2)
ssr = sum((y_pred .- y_test).^2)
sst = sum((y_test .- mean(y_test)).^2)
r2 = 1.0 - ssr / sst

# plot the testing result
actual_values = y_test
predicted_values = y_pred
plt_test = scatter(1:length(actual_values), actual_values, label="Actual", legend=:topleft, markersize=7)
scatter!(1:length(predicted_values), predicted_values, label="Predicted", markersize=7)
xlabel!("Data Point")
ylabel!("Value")
title!("Actual vs. Predicted Data Points")
savefig(plt_test, "test_$train_id.pdf")

# save the evaluation parameters
metricsfile = open("metrics_$train_id.txt", "w")
println(metricsfile, "mean_test: ", mean(y_test))
println(metricsfile, "MSE_test_ave: $mse")
println(metricsfile, "R-squared_test: $r2")
close(metricsfile)