#= calculate the distribution possibilities for plasmid clusters.
       Julia v1.9.1
       CSV v0.10.11
       DataFrames v1.6.1
       StatsBase v0.33.21
       DataStructures v0.18.15
       Statistics v1.9.0
=# 

# The output of step 1 was used to generate "APntax.csv" and "Pn.csv" in Excel.

# step 1: for a given cluster, number of plasmids of each taxon was calculated for taxonomic rank Genus, Family, Order, Class, and Phylum, respectively.

using CSV, DataFrames, StatsBase

# read in the starting file "plasmid_tax.csv"
df = CSV.read("plasmid_tax.csv", DataFrame, header=true)

# separate the DataFrame by the "Groups" column and specify the column names that will be used for counting
grouped_df = groupby(df, :Groups)
column_id = :Groups
column_phylum = :phylum
column_class = :class
column_order = :order
column_family = :family
column_genus = :genus

# create the output file "distribution_count.txt" and write the header
txtfile = open("distribution_count.txt", "w")
println(txtfile, "Clusters", "\t", "Total_num", "\t", "Phylum", "\t", "Class", "\t", "Order", "\t", "Family", "\t", "Genus")
close(txtfile)

# count the plasmid numbers in each taxonomic rank for each cluster
for group in grouped_df
       # extract taxonomy info
       words_total = collect(Iterators.flatten(split.(string.(group[!,column_id]), r"\W+")))
       words_phylum = collect(Iterators.flatten(split.(lowercase.(group[!,column_phylum]), r"\W+")))
       words_class = collect(Iterators.flatten(split.(lowercase.(group[!,column_class]), r"\W+")))
       words_order = collect(Iterators.flatten(split.(lowercase.(group[!,column_order]), r"\W+")))
       words_family = collect(Iterators.flatten(split.(lowercase.(group[!,column_family]), r"\W+")))
       words_genus = collect(Iterators.flatten(split.(lowercase.(group[!,column_genus]), r"\W+")))
       total_counts = countmap(words_total)
       phylum_counts = countmap(words_phylum)
       class_counts = countmap(words_class)
       order_counts = countmap(words_order)
       family_counts = countmap(words_family)
       genus_counts = countmap(words_genus)

       # save the results
       txtfile = open("distribution_count.txt", "a")
       global txtfile

       for (k, v) in total_counts
              print(txtfile, k, "\t", v)
       end
       print(txtfile,"\t")
       for (k, v) in phylum_counts
              print(txtfile, k, ",", v, ";")
       end
       print(txtfile,"\t")
       for (k, v) in class_counts
              print(txtfile, k, ",", v, ";")
       end
       print(txtfile,"\t")
       for (k, v) in order_counts
              print(txtfile, k, ",", v, ";")
       end
       print(txtfile,"\t")
       for (k, v) in family_counts
              print(txtfile, k, ",", v, ";")
       end
       print(txtfile,"\t")
       for (k, v) in genus_counts
              print(txtfile, k, ",", v, ";")
       end
       println(txtfile,"\t")
       close(txtfile)
end

# step 2: normalization of the plasmid number of each taxon

# "APntax.csv" contains the the average plasmid number per cluster (APntax) for a given taxon. "Pn.csv" has the raw plasmid number (Pn) of each taxon
df_ave = CSV.read("APntax.csv", DataFrame, header=true)
df_cls = CSV.read("Pn.csv", DataFrame, header=true)

# covert df_cls to dictionary
dic_cls = Dict()
for row in eachrow(df_cls)
       main_key = row[1]
       sub_dict = Dict()
       for i in 2:2:ncol(df_cls)-1
           sub_dict[row[i]] = row[i+1]
       end
       dic_cls[main_key] = sub_dict
end

# remove "missing" from dic_cls
for key in keys(dic_cls)
       delete!(dic_cls[key], missing)
end

# covert df_ave to dictionary
dic_ave = Dict()
for row in eachrow(df_ave)
       dic_ave[row[1]] = row[4]
end

# normalization
for key in keys(dic_cls)
       temp_dic = dic_cls[key]
       for k in keys(temp_dic)
              Nave = dic_ave[k]
              if temp_dic[k] >= Nave
                     temp_dic[k] = 1
                     else
                            temp_dic[k] = temp_dic[k] / Nave
                     end
       end
end
          
# save result to "NPntax.txt"
txtfile = open("NPntax.txt", "w")
for key in keys(dic_cls)
       temp_dic = dic_cls[key]
       print(txtfile, key, "\t")
       for (k, v) in temp_dic
              print(txtfile, k, "\t", v, "\t")
       end
       println(txtfile)
end
close(txtfile)

# "NPntax.txt" was converted to CSV file "NPntax.csv" in Excel.

# step 3: calculate the distribution score (DSrank) for a given rank

using DataStructures, Statistics

# "NPntax.csv" contains the normalized plasmid number of each taxon
df_norm = CSV.read("NPntax.csv", DataFrame, header=true)
dic_norm = Dict()
for row in eachrow(df_norm)
       main_key = row[1]
       sub_dict = Dict()
       for i in 2:2:ncol(df_norm)-1
           sub_dict[row[i]] = row[i+1]
       end
       dic_norm[main_key] = sub_dict
end
# remove "missing" from dic_norm
for key in keys(dic_norm)
       delete!(dic_norm[key], missing)
end

# group the values according to taxonomic ranks for each cluster
dic_group = Dict()
for (key, subdict) in dic_norm
    nested_subdict = Dict()
    for (subkey, value) in subdict
        subkey_prefix = split(subkey, "_")[1]
        if !haskey(nested_subdict, subkey_prefix)
            nested_subdict[subkey_prefix] = []
        end
        push!(nested_subdict[subkey_prefix], value)
    end
    dic_group[key] = nested_subdict
end

# order the sub_dicts in dic_group to "Genus, Family, Order, Class, Phylum"
key_order = ["genus", "family", "order", "class", "phylum"]
dic_group_sort = Dict()
for k in keys(dic_group)
    sorted_subdict = OrderedDict()
    for key in key_order
       if haskey(dic_group[k], key)
              sorted_subdict[key] = dic_group[k][key]
       end
    end
    dic_group_sort[k] = sorted_subdict
end

# use dic_group_sort to calculate the plasmid distribution score (DSrank) for each taxonomic rank.
dic_score = Dict()
for keys in keys(dic_group_sort)
    subscores = OrderedDict()
    for (k, v) in dic_group_sort[keys]
        std_value = std(v)
        if isnan(std_value)
            std_value = 1
        end
        score = (1-std_value) * length(v)
        subscores[k] = score
    end
    dic_score[keys] = subscores
end

# step 4: calculate the distribution possibility
# use dic_score to calculate the plasmid distribution score (DSrank)
dic_P = Dict()
dic_P = copy(dic_score)
for (key, subdic) in dic_P
    for (k, v) in subdic
        if v >= 2
            subdic[k] = 1
        else
            subdic[k] = v/2
        end
    end
end

# save the result to "score_matrix.txt" for examination purpose
txtfile = open("score_matrix.txt", "w")
println(txtfile, "Cls", "\t", "genus_possibility", "\t", "family_possibility", "\t", "order_possibility", "\t", "class_possibility", "\t", "phylum_possibility", "\t")
for key in keys(dic_P)
       temp_dic = dic_P[key]
       print(txtfile, key, "\t")
       for (k, v) in temp_dic
              print(txtfile, k, "\t", v, "\t")
       end
       println(txtfile)
end
close(txtfile)

# finally, calculate the sum of distribution possibility scores (Psum) in dic_P
dic_Pc = Dict()
for (key, subdic) in dic_P
    v_sum = 0
    for (k, v) in subdic
        v_sum = v_sum + v
    end
    dic_Pc[key] = v_sum
end

# Save the results to "Psum.txt"
txtfile = open("Psum.txt", "w")
println(txtfile, "Cls", "\t", "Psum")
for (k, v) in dic_Pc
    print(txtfile, k, "\t", v, "\t")
    println(txtfile)
end
close(txtfile)
