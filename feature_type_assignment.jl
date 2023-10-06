#= feature importance analysis: assign feature types to features in interested combinations
	Julia v1.9.1
       CSV v0.10.11
       DataFrames v1.6.1
=#

using DataFrames
using CSV

# "feature_combinations.csv" contains feature combinations that we want to assign feature types. "all_features.csv" contains feature names and the corresponding type categories.
df_combs = CSV.read("feature_combinations.csv", DataFrame)
combs = df_combs[!, 1]
df_types = CSV.read("all_features.csv", DataFrame)
# store types to dictionary
dic_types = Dict()
for row in eachrow(df_types)
       dic_types[row[1]] = row[2]
end

# store the results
historyfile = open("feature_types.txt", "w")
println(historyfile, "combinations", "\t", "types")
close(historyfile)

for row in combs
# split the string at commas and remove leading/trailing spaces
       cols = split(row, ",")
       cols = strip.(cols)
       tp_vec =[] # store the feature types
       for col in cols
              tp = dic_types[col]
              push!(tp_vec, tp)
       end
       historyfile = open("feature_types.txt", "a")
       println(historyfile, cols, "\t", tp_vec)
       close(historyfile)
end