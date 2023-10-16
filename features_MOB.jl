#= prepare feature replicon type(s), relaxase type(s), mate-pair formation type, and origin of transfer type.
Need to change file_name at line 12 before started.
       Julia v1.9.1
       CSV v0.10.11
       DataFrames v1.6.1
       StatsBase v0.33.21
=# 
using DataFrames
using CSV
using StatsBase

file_name = "repT" # change to target feature: repT (replicon type), relaxaseT (relaxase type), mpfT (mate-pair formation type), oriT (origin of transfer type)

df = CSV.read("$file_name.csv", DataFrame)
# group the DataFrame by the "groups" column
grouped_df = groupby(df, :Groups)

nor_dic = Dict() # store the calculated percentage for each group
for group in grouped_df
    group_name = unique(group[!, 1])
    values = group[!, 2]
    value_counts = countmap(values)
    for k in keys(value_counts)
        delete!(value_counts, missing)
    end
    sum_v = 0
    for (k, v) in value_counts
        sum_v += v
    end
    for (k, v) in value_counts
        value_counts[k] = round(100 * v / sum_v)
    end
    nor_dic[group_name] = value_counts
end

# save the results
df_rep = CSV.File("$file_name-ID.csv", header=false) |> DataFrame # corresponding files containing all of the unique categorical values of a feature: repT-ID.csv, relaxaseT-ID.csv, mpfT-ID.csv, oriT-ID.csv
subfeatures = Matrix(df_rep)
textfile = open("$file_name.txt", "w")

print(textfile, "group", "\t")
for i in subfeatures
    print(textfile, i, "\t")
end
println(textfile)
for (key, subdic) in nor_dic
    print(textfile, key, "\t")
    for i in subfeatures
        if i in keys(subdic)
            print(textfile, subdic[i], "\t")
        else
            print(textfile, 0, "\t")
        end
    end
    println(textfile)
end
close(textfile)
