#= feature importance analysis: fix "smallest" to 0 and then randomly remove features in combinations of 16 to 20 features. 
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
using Random
using Resample
using ProgressMeter

run_id = "run_id"
draw_num=50000 # luck guess rounds
min_comb=16 # min number of elements in combination
max_comb=20 # max number of elements in combination
standard = 0.8 # starting cutoff of MSE of testing dataset
feature_num = 16 # starting number of removed features. We want to get big effect with as little features as possible 

# define a function to replace values of one column with 0
function feature_zero(data_df::DataFrame, col_name)
       modified_df = copy(data_df)
       modified_df[!, col_name] .= 0.0
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

# "feature_matrix_F.csv" is the data matrix containing target Psum and the features
df_origin = CSV.read("feature_matrix_F.csv", DataFrame)
df_names = CSV.read("feature_names.csv", DataFrame) # supply all feature names except for "smallest"
col_names = df_names[!, 1]

# store the running history into txt file
historyfile = open("Luck_$run_id.txt", "w")
println(historyfile, "removed_feature_name", "\t", "MSE_test", "\t", "r2_test", "\t",  "MSE_test(P>1)")
close(historyfile)

# fix "smallest" to 0 and split the dataset
df_origin_new = feature_zero(df_origin, "smallest")
train, df_test = partition(df_origin_new, 0.8, shuffle=true, rng=479)
df_test2 = filter(row -> row.sum_of_P > 1, df_test)
df_minor = filter(row -> row.sum_of_P > 1.2, train)
df_minor2 = filter(row -> row.sum_of_P > 2.5, train)
num_minor = nrows(df_minor)
num_minor2 = nrows(df_minor2)
newdata = smote(df_minor, floor(Int64, round(num_minor*2)))
newdata2 = smote(df_minor2, floor(Int64, round(num_minor2*1)))
df_train = vcat(train, DataFrame(newdata), DataFrame(newdata2))

# randomly remove features
tested_combinations = [] # record the tested combinations to avoid redundant testing
@showprogress for i in 1:draw_num
       seed = rand(min_comb:max_comb, 1)
       index = randperm(142)[1:seed[1]]
       sort!(index)
       !(index in tested_combinations) || continue
       push!(tested_combinations, index)
       # remove the features for testing
       col = col_names[index]
       df_train_mod = feature_zero(df_train, col)
       df_test_mod = feature_zero(df_test, col)
       df_test2_mod = feature_zero(df_test2, col)
       X = df_train_mod[!, 2:end]
       y = df_train_mod[!, 1]
       X_test = df_test_mod[!, 2:end]
       y_test = df_test_mod[!, 1]
       X_test2 = df_test2_mod[!, 2:end]
       y_test2 = df_test2_mod[!, 1]
       
       # create a machine and train model on the modified dataset
       stack_mach = machine(stack_model, X, y)
       MLJ.fit!(stack_mach, force=true, verbosity=0)

       y_pred = MLJ.predict_mode(stack_mach, X_test)
       y_pred2 = MLJ.predict_mode(stack_mach, X_test2)
       mse = mean((y_pred .- y_test).^2)
       mse2 = mean((y_pred2 .- y_test2).^2)
       ssr = sum((y_pred .- y_test).^2)
       sst = sum((y_test .- mean(y_test)).^2)
       r2 = 1.0 - ssr / sst

       # record combinations that increase mse; if the new MSE is similar with previous value, the one with less features_removal is kept.
       if abs(mse - standard) < 0.02 && length(col) < feature_num
              standard = mse
              feature_num = length(col)
              historyfile = open("Luck_$run_id.txt", "a")
              println(historyfile, col, "\t", mse, "\t", r2, "\t", mse2)
              close(historyfile)
              elseif (mse - standard) > 0.02
                     standard = mse
                     feature_num = length(col)
                     historyfile = open("Luck_$run_id.txt", "a")
                     println(historyfile, col, "\t", mse, "\t", r2, "\t", mse2)
                     close(historyfile)
              end
end
