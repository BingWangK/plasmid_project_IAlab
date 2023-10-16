#= feature importance analysis: randomly remove features in combinations of 7 features. Feature types include "beneficial traits", "counter-defense", "DNA binding", "inducer binding", "mobility", "plasmid backbone", "plasmid maintenance", "P-loop_NTPase", "Rossmann", "toxin-antitoxin", "transmembrane transporters", "unkown".
	Julia v1.9.1
	MLJ v0.19.2
	DataFrames v1.6.1
        CSV v0.10.11
        MLJEnsembles v0.3.3
	Random
        Resample v1.0.2
        ProgressMeter v1.7.2
=#

using MLJ
using DataFrames
using CSV
using MLJEnsembles
using Random
using Resample
using ProgressMeter

run_id = "run_id"
draw_num = 11000 # number of iterations
standard = 0.7 # starting cutoff of MSE of testing dataset

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
decided = decision(max_depth = 23, min_samples_split = 4, n_subfeatures = 70, min_samples_leaf = 10)
evoed = evo(loss=:tweedie, nrounds = 70, gamma = 5.1, eta = 0.5, max_depth =7, colsample = 0.822, nbins = 18)
lighter = light(num_iterations = 200, max_depth = 5, feature_fraction = 0.9, learning_rate = 0.055,min_data_in_leaf = 25, lambda_l1 = 2, min_gain_to_split = 1, time_out = 14400)

stack_model = Stack(;metalearner=learner,
                measures=rms,
                resampling=CV(nfolds=5, shuffle = true),
                model1 = decided,
                model2 = evoed,
                model3 = lighter
                )

# "feature_matrix_F3.csv" is the data matrix containing the Psum and the values for 140 features.
df_origin = CSV.read("feature_matrix_F3.csv", DataFrame)
df_names = CSV.read("all_features.csv", DataFrame)
col_names = df_names[!, 1]

# get a baseline with only dropping "smallest", and use the same split to do initial screening
train_init, df_test_init = partition(df_origin, 0.8, shuffle=true, rng=479)
df_minor_init = filter(row -> row.Psum > 1.2, train_init)
df_minor2_init = filter(row -> row.Psum > 2.5, train_init)
num_minor_init = nrows(df_minor_init)
num_minor2_init = nrows(df_minor2_init)
newdata_init = smote(df_minor_init, floor(Int64, round(num_minor_init*2)))
newdata2_init = smote(df_minor2_init, floor(Int64, round(num_minor2_init*1)))
df_train_init = vcat(train_init, DataFrame(newdata_init), DataFrame(newdata2_init))
df_train_start = feature_zero(df_train_init, "smallest")
df_test_start = feature_zero(df_test_init, "smallest")
X_init = df_train_start[!, 2:end]
y_init = df_train_start[!, 1]
X_test_init = df_test_start[!, 2:end]
y_test_init = df_test_start[!, 1]
mach_init = machine(stack_model, X_init, y_init)
MLJ.fit!(mach_init, force=true, verbosity=0) 
y_pred_init = MLJ.predict_mode(mach_init, X_test_init)
mse_init = mean((y_pred_init .- y_test_init).^2)

# store the running history into txt file
runlog = open("runlog_$run_id.txt", "w")
println(runlog, "iterations", "\t", "combinations")
close(runlog)

# save the combinations with MSE > standard
MSEfile = open("MSE_$run_id.txt", "w")
println(MSEfile, "tested_combinations", "\t", "MSE_test", "\t", "MSE_test_std", "\t", "R2_test", "\t", "R2_test_std", "\t", "MSE_test(P>1)", "\t", "MSE_test(P>1)_std")
close(MSEfile)

# randomly remove feature combinations
@showprogress for i in 1:draw_num
       seed = randperm(140)[1:7]
       sort!(seed)
       col = col_names[seed]

       runlog = open("runlog_$run_id.txt", "a")
       println(runlog, i, "\t", col)
       close(runlog)
       
       # drop columns for testing
       df_train_mod_comb = feature_zero(df_train_init, col)
       df_test_mod_comb = feature_zero(df_test_init, col)
       X_comb = df_train_mod_comb[!, 2:end]
       y_comb = df_train_mod_comb[!, 1]
       X_test_comb = df_test_mod_comb[!, 2:end]
       y_test_comb = df_test_mod_comb[!, 1]
       stack_mach = machine(stack_model, X_comb, y_comb)
       MLJ.fit!(stack_mach, force=true, verbosity=0)
       y_pred_comb = MLJ.predict_mode(stack_mach, X_test_comb)
       mse_comb = mean((y_pred_comb .- y_test_comb).^2)
              
       if mse_comb > mse_init
              df_origin_mod = feature_zero(df_origin, col)
              mse_test_vec = []
              R2_test_vec = []
              mse_test2_vec = []
              for j in 1:100
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
                     push!(R2_test_vec, r2)
                     push!(mse_test2_vec, mse2)
              end
              # calculate mean and standard deviation
              mse_test_ave = mean(mse_test_vec)
              if mse_test_ave > standard
                     mse_test_std = std(mse_test_vec)
                     r2_test_ave = mean(R2_test_vec)
                     r2_test_std = std(R2_test_vec)
                     mse_test2_ave = mean(mse_test2_vec)
                     mse_test2_std = std(mse_test2_vec)
                     MSEfile = open("MSE_$run_id.txt", "a")
                     println(MSEfile, col, "\t", mse_test_ave, "\t", mse_test_std, "\t", r2_test_ave, "\t", r2_test_std, "\t", mse_test2_ave, "\t", mse_test2_std)
                     close(MSEfile)
              end
       end
end
