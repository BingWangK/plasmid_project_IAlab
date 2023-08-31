#= model stack: searching combinations of the base models
	Julia v1.9.1
	MLJ v0.19.2
	DataFrames v1.6.1
       CSV v0.10.11
       MLJEnsembles v0.3.3
       Combinatorics v1.0.2
	Random
       Resample v1.0.2
=#

using MLJ
using DataFrames
using CSV
using MLJEnsembles
using Combinatorics
using Random
using Resample

# "feature_matrix_F.csv" is the data matrix containing target Psum and the features
df_origin = CSV.read("feature_matrix_F.csv", DataFrame)

# load meta-learner
meta = @MLJ.load LinearRegressor pkg=GLM
learner = meta(fit_intercept = true, dropcollinear=true)

# load base models
cat = @MLJ.load CatBoostRegressor pkg=CatBoost
evo = @MLJ.load EvoTreeRegressor pkg=EvoTrees #v0.14.11
random = @MLJ.load RandomForestRegressor pkg=DecisionTree
decision = @MLJ.load DecisionTreeRegressor pkg=DecisionTree
light = @MLJ.load LGBMRegressor pkg=LightGBM

tiger=cat(loss_function="Quantile", iterations=1500, depth=6, learning_rate=0.1, l2_leaf_reg=1, border_count=10)
forest=random(n_trees = 130, n_subfeatures = 70, max_depth = 31, min_samples_split =2)
decided=decision(max_depth = 23, min_samples_split = 4, n_subfeatures = 70, min_samples_leaf = 10)
evoed=evo(loss=:tweedie, nrounds = 70, gamma = 5.1, eta = 0.5, max_depth =7, colsample = 0.822, nbins = 18)
lighter = light(num_iterations = 100, max_depth = 5, feature_fraction = 0.8, learning_rate = 0.055,min_data_in_leaf = 25, lambda_l1 = 1, min_gain_to_split = 1, time_out = 14400)

# generate the combinations
base_list = [tiger, evoed, forest, decided, lighter]
base_number = [1, 2, 3, 4, 5]
combination = collect(combinations(base_number))

# logging the performance on testing dataset
log = open("combinations_LinearRegressor.txt", "w")
println(log, "MSE", "\t", "base_num", "\t", "base_models")
close(log)

comb_standard = 1
for comb in combination
       base_models = base_list[comb]
       L = length(base_models)
       L > 1 || continue # rule out single base model situation
       if L == 2
              stack_model = Stack(;metalearner=learner,
              measures=rms,
              resampling=CV(nfolds=5, shuffle = true),
              model1=base_models[1],
              model2=base_models[2])
              elseif L == 3
                     stack_model = Stack(;metalearner=learner,
                     measures=rms,
                     resampling=CV(nfolds=5, shuffle = true),
                     model1=base_models[1],
                     model2=base_models[2],
                     model3=base_models[3])
              elseif L == 4
                     stack_model = Stack(;metalearner=learner,
                     measures=rms,
                     resampling=CV(nfolds=5, shuffle = true),
                     model1=base_models[1],
                     model2=base_models[2],
                     model3=base_models[3],
                     model4=base_models[4])
                     elseif L == 5
                            stack_model = Stack(;metalearner=learner,
                            measures=rms,
                            resampling=CV(nfolds=5, shuffle = true),
                            model1=base_models[1],
                            model2=base_models[2],
                            model3=base_models[3],
                            model4=base_models[4],
                            model5=base_models[5])
                     end
     
       # train a model stack 5 times by splitting dataset into training and testing datasets
       mse_vec = []
       for i in 1:5
              train, df_test = partition(df_origin, 0.8, shuffle=true, rng=479)
              df_minor = filter(row -> row.sum_of_P > 1.2, train)
              df_minor2 = filter(row -> row.sum_of_P > 2.5, train)
              num_minor = nrows(df_minor)
              num_minor2 = nrows(df_minor2)
              newdata = smote(df_minor, floor(Int64, round(num_minor*2)))
              newdata2 = smote(df_minor2, floor(Int64, round(num_minor2*1)))
              df_train = vcat(train, DataFrame(newdata), DataFrame(newdata2))
              X = df_train[!, 2:end]
              y = df_train[!, 1]
              X_test = df_test[!, 2:end]
              y_test = df_test[!, 1]

              mach = machine(stack_model, X, y)
              MLJ.fit!(mach, force=true)

              y_pred = MLJ.predict_mode(mach, X_test)
              mse = mean((y_pred .- y_test).^2)
              push!(mse_vec, mse) 
       end
       # record the mean of MSE of the 5 times training
       mse_ave = mean(mse_vec)
       if mse_ave < comb_standard
              comb_standard = mse_ave
              log = open("combinations_LinearRegressor.txt", "a")
              println(log, comb_standard, "\t", L, "\t", base_models)
              close(log)
       end
end
