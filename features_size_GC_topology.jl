#= prepare feature size, GC, topology.
Need to change file_name at line 13 and header at lines 40 - 53 before started.
       Julia v1.9.1
       CSV v0.10.11
       DataFrames v1.6.1
       StatsBase v0.33.21
=# 

using DataFrames
using CSV
using StatsBase

file_name = "size" # change to target feature: size or GC or topology

df = CSV.read("$file_name.csv", DataFrame)
# group the DataFrame by the "groups" column
grouped_df = groupby(df, :Groups)
nor_dic = Dict() # store the calculated percentage for all groups
for group in grouped_df # calculate the percentage of each categorical value for each cluster
    group_name = unique(group[!, 1])
    values = group[!, 2]

    # get a dictionary storing numbers of each categorical value
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
# headers for size conversion
subfeatures = ["smallest", "smaller", "small", "average", "large", "larger","largest"]
textfile = open("$file_name.txt", "w")
println(textfile, "group", "\t", "smallest", "\t", "smaller", "\t", "small", "\t", "average", "\t", "large", "\t", "larger", "\t", "largest")

# headers for GC conversion
#subfeatures = ["lowest", "lower", "low", "mean", "high", "higher","highest"]
#textfile = open("$file_name.txt", "w")
#println(textfile, "group", "\t", "lowest", "\t", "lower", "\t", "low", "\t", "mean", "\t", "high", "\t", "higher", "\t", "highest")

# headers for topology conversion
#subfeatures = ["circular", "linear"]
#textfile = open("$file_name.txt", "w")
#println(textfile, "group", "\t", "circular", "\t", "linear")

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
