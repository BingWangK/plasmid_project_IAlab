#= feature importance analysis: test features one by one.
	Julia v1.9.1
	MLJ v0.19.2
	DataFrames v1.6.1
       CSV v0.10.11
       MLJEnsembles v0.3.3
       ProgressMeter v1.7.2
	Random
       Resample v1.0.2
=#

using MLJ
using DataFrames
using CSV
using MLJEnsembles
using ProgressMeter
using Random
using Resample

run_id = "test"

# define a function to drop the testing feature
function feature_zero(data_df::DataFrame, col_name)
       df_copy = copy(data_df)
       modified_df = df_copy[!, Not(col_name)]
       return modified_df
   end

# load meta-learner
meta = @MLJ.load LinearRegressor pkg=GLM
learner = meta(fit_intercept = false, dropcollinear = false)
# load base models
evo = @MLJ.load EvoTreeRegressor pkg=EvoTrees #v0.14.11
decision = @MLJ.load DecisionTreeRegressor pkg=DecisionTree
light = @MLJ.load LGBMRegressor pkg=LightGBM
decided=decision(max_depth = 23, min_samples_split = 4, n_subfeatures = 70, min_samples_leaf = 10)
evoed=evo(loss=:tweedie, nrounds = 70, gamma = 5.1, eta = 0.5, max_depth =7, colsample = 0.822, nbins = 18)
lighter = light(num_iterations = 200, max_depth = 5, feature_fraction = 0.9, learning_rate = 0.055,min_data_in_leaf = 25, lambda_l1 = 2, min_gain_to_split = 1, time_out = 14400)

stack_model = Stack(;metalearner=learner,
                measures=rms,
                resampling=CV(nfolds=5, shuffle = true),
                model1 = decided,
                model2 = evoed,
                model3 = lighter
                )

# "feature_matrix_F3.csv" is the data matrix containing target Psum and the features
df_origin = CSV.read("feature_matrix_F3.csv", DataFrame)

# store the results
historyfile = open("feature_analysis_$run_id.txt", "w")
println(historyfile, "removed_feature_name", "\t", "MSE_test", "\t", "MSE_test_std", "\t", "r2_test", "\t", "r2_test_std", "\t", "MSE_test(P>1)", "\t", "MSE_test(P>1)_std")
close(historyfile)

# get baseline
mse_vec = []
r2_vec = []
mse2_vec = []

# split the dataset 100 times and training/testing the model stack
@showprogress for i in 1:100
       train, df_test = partition(df_origin, 0.8, shuffle=true, rng=479)
       df_test2 = filter(row -> row.Psum > 1, df_test) # test the performance on the minority part
       df_minor = filter(row -> row.Psum > 1.2, train)
       df_minor2 = filter(row -> row.Psum > 2.5, train)
       num_minor = nrows(df_minor)
       num_minor2 = nrows(df_minor2)
       newdata = smote(df_minor, floor(Int64, round(num_minor*2)))
       newdata2 = smote(df_minor2, floor(Int64, round(num_minor2*1)))
       df_train = vcat(train, DataFrame(newdata), DataFrame(newdata2))
       X = df_train[!, 2:end]
       y = df_train[!, 1]
       X_test = df_test[!, 2:end]
       y_test = df_test[!, 1]
       X_test2 = df_test2[!, 2:end]
       y_test2 = df_test2[!, 1]
       
       mach = machine(stack_model, X, y)
       MLJ.fit!(mach, force=true, verbosity=0)
       
       y_pred = MLJ.predict_mode(mach, X_test)
       y_pred2 = MLJ.predict_mode(mach, X_test2)
       mse = mean((y_pred .- y_test).^2)
       mse2 = mean((y_pred2 .- y_test2).^2)
       ssr = sum((y_pred .- y_test).^2)
       sst = sum((y_test .- mean(y_test)).^2)
       r2 = 1.0 - ssr / sst
       
       push!(mse_vec, mse)
       push!(mse2_vec, mse2)
       push!(r2_vec, r2)
end
       
mse_ave = mean(mse_vec)
mse_std = std(mse_vec)
mse2_ave = mean(mse2_vec)
mse2_std = std(mse2_vec)
r2_ave = mean(r2_vec)
r2_std = std(r2_vec)
       
# save the results
historyfile = open("feature_analysis_$run_id.txt", "a")
println(historyfile, "baseline", "\t", mse_ave, "\t", mse_std, "\t", r2_ave, "\t", r2_std, "\t", mse2_ave, "\t", mse2_std)
close(historyfile)

# test features by removing them one by one
# get all column names except for column of Psum
col_names = names(df_origin[!, 2:end])

@showprogress for col in col_names
       # drop the tested column
       df_origin_mod = feature_zero(df_origin, col)

       mse_test_vec = []
       r2_test_vec = []
       mse_test2_vec = []
       for i in 1:100
              train, df_test = partition(df_origin_mod, 0.8, shuffle=true, rng=479)
              df_test2 = filter(row -> row.Psum > 1, df_test)
              df_minor = filter(row -> row.Psum > 1.2, train)
              df_minor2 = filter(row -> row.Psum > 2.5, train)
              num_minor = nrows(df_minor)
              num_minor2 = nrows(df_minor2)
              newdata = smote(df_minor, floor(Int64, round(num_minor*2)))
              newdata2 = smote(df_minor2, floor(Int64, round(num_minor2*1)))
              df_train = vcat(train, DataFrame(newdata), DataFrame(newdata2))
              X = df_train[!, 2:end]
              y = df_train[!, 1]
              X_test = df_test[!, 2:end]
              y_test = df_test[!, 1]
              X_test2 = df_test2[!, 2:end]
              y_test2 = df_test2[!, 1]
       
              mach = machine(stack_model, X, y)
              MLJ.fit!(mach, force=true, verbosity=0)
       
              y_pred = MLJ.predict_mode(mach, X_test)
              y_pred2 = MLJ.predict_mode(mach, X_test2)
              mse = mean((y_pred .- y_test).^2)
              mse2 = mean((y_pred2 .- y_test2).^2)
              ssr = sum((y_pred .- y_test).^2)
              sst = sum((y_test .- mean(y_test)).^2)
              r2 = 1.0 - ssr / sst
              push!(mse_test_vec, mse)
              push!(r2_test_vec, r2)
              push!(mse_test2_vec, mse2)
       end

       # calculate mean and standard deviation
       mse_test_ave = mean(mse_test_vec)
       mse_test_std = std(mse_test_vec)
       r2_test_ave = mean(r2_test_vec)
       r2_test_std = std(r2_test_vec)
       mse_test2_ave = mean(mse_test2_vec)
       mse_test2_std = std(mse_test2_vec)
       historyfile = open("feature_analysis_$run_id.txt", "a")
       println(historyfile, col, "\t", mse_test_ave, "\t", mse_test_std, "\t", r2_test_ave, "\t", r2_test_std, "\t", mse_test2_ave, "\t", mse_test2_std)
       close(historyfile)
end
