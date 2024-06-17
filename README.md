# plasmid_project_IAlab
This repository contains the necessary code for repeating the results described in the publication "Machine Learning Suggests That Small Size Helps Broaden Plasmid Host Range" (https://doi.org/10.3390/genes14112044). Folder "data_file" contains the input files for the scripts.

The ANI edgeweight file for plasmid clustering analysis, plasmid network files in Cytoscape format, and the final data matrix for model training and testing are deposited to DRYAD (https://doi.org/doi:10.5061/dryad.1g1jwsv31).

# STEPS:
1. Plasmid sequences were collected following “_2.1. Plasmid Sequences Collection_” of the Materials and Methods of the publication.

2. Following “_2.2. Plasmid Clustering_” of the publication to the step of equation (1). To apply equation (1) to the ANI values, we used awk:

```
$ awk 'BEGIN{FS=OFS="\t"} {if ($3 >= 95) $3 = $3/100; else $3 = 1 / (1 + 20 * (1 - $3/100))} 1' fastANI_input > fastANI_edgeweights #column 3 of input file has the ANI values, “fastANI_edgeweights” is the output.
```

The “fastANI_edgeweights” file can be downloaded from DRYAD (https://doi.org/doi:10.5061/dryad.1g1jwsv31). The downloaded zip file contains (1) “fastANI_edgeweights”; (2) “plasmid_net_connected.cys”: The original plasmid network; (3) “plasmid_net_grouped_MOB.cys”: The clusters were separated based on clusters (Fig. 2 of the publication). (4) “feature_matrix_F3.csv”: The final data matrix for machine learning model training and testing.

3. Plasmids clustering was performed using the Leiden algorithm with “fastANI_edgeweights” and “leidenalg_CPM.py”. We used python3.7-conda4.5 environment to run the program. The resulting “plasmid_clusters.gml” file contains the network graph and cluster membership. The “clusters_output.txt” records the extracted cluster membership from the “plasmid_clusters.gml” file. Clusters with < 3 members are removed.

4. The “plasmid_clusters.gml” can be visualized with Cytoscape. However, to facilitate metadata mapping, the “fastANI_edgeweights” was loaded into Cytoscape, metadata from “cytoscape_metadata.txt” was imported, and plasmids not in a cluster were removed. This created the original plasmid network “plasmid_net_connected.cys”. 

5. To create “plasmid_net_grouped_MOB.cys” (Fig. 2 of the publication). 
Use below command to keep edges between nodes from the same cluster:
```
$ awk 'NR == FNR { group[$1] = $2; next } (group[$1] == group[$2]) { print }' cluster_output.txt fastANI_edgeweights > fastANI_edgeweights_grouped
```

Then load "fastANI_edgeweights_grouped" into Cytoscape and map “MOB_metadata.txt”. if one wish to do more analysis, “cytoscape_metadata.txt” can also be mapped. This creates "plasmid_net_grouped_MOB.cys".

6. Calculate the distribution possibility for each cluster using “distribution_possibility.jl”. The output “Psum.txt” contains the final calculated distribution possibilities, which will be used as the output vector in the model training.

7. Prepare the input vector (“_2.4. Extraction of Features_” of the publication). (1) Conversion of categorical values of the features to the fraction of the total for each cluster using “features_size_GC_topology.jl" and "feature_MOB.jl”. (2) Calculate the percentage distribution of the pfam models in each cluster using “features_pfamscan.sh”. (3) All converted features were combined into “feature_matrix_A.csv”. (4) Scale all features to 0 - 1 and remove features with < 1% variation using “feature_matrix_processing.jl", The output will be "feature_matrix_F.csv". After first-round of model building, the feature_matrix_F.csv was updated to feature_matrix_F3.csv (Please see the corresponding Method section of the publication).
   
8. Hyperparameters tuning for individual base models using “regressor_tuning.jl”.

9. After tunning individual models, we searched for the best base model combinations using “model_stack_searching.jl”.

10. Feature importance analysis by dropping column method: “feature_analysis_single.jl”, “feature_analysis_double.jl”, “feature_analysis_Luck.jl”, “feature_analysis_Luck7.jl”.

11. Assign feature types to features in interested combinations: “feature_type_assignment.jl”.


| Files | Description |
| --- | --- |
| leidenalg_CPM.py | Plasmid clustering by using leidenalg with CPM |
| cytoscape_metadata.txt | Metadata of plasmid network |
| MOB_metadata.txt | Mobility types predicted by MOB-suite (https://github.com/phac-nml/mob-suite) |
| clusters_output.txt | Output of “leidenalg_CPM.py”. Because the cluster numbering is random, to be consistent with the published data, we provided this output file |
| distribution_possibility.jl | Calculate distribution possibility for each plasmid cluster|
| plasmid_tax.csv | Containing taxonomic lineage information, which will be used in step 1 of "distribution_possibility.jl" |
| APntax.csv and Pn.csv | "APntax.csv" contains the the average plasmid number per cluster (APntax) for a given taxon. "Pn.csv" has the raw plasmid number (Pn) of each taxon, which will be used in step 2 of "distribution_possibility.jl" |
| NPntax.csv | "NPntax.csv" contains the normalized plasmid number of each taxon, which will be used in step 3 of "distribution_possibility.jl" |
| features_size_GC_topology.jl, feature_MOB.jl | Conversion of categorical values of the features to fraction of total for each cluster |
| size.csv, GC.csv, topology.csv, repT-ID.csv, repT.csv, relaxaseT-ID.csv, relaxaseT.csv, mpfT-ID.csv, mpfT.csv, oriT-ID.csv, oriT.csv | The original categorical values of these features |
| features_pfamscan.sh | Calculate the percentage of the pfam_models in each cluster |
| pfamscan_CL.txt | It contains the file names that need to be processed by features_pfamscan.sh |
| pfamscan_CL.zip | A total of 1125 files corresponding to the 1125 clusters. The predicted pfam_models are listed in each file |
| feature_matrix_processing.jl | To scale all features to 0 - 1 and remove features with < 1% variation |
| feature_matrix_A.csv | The combined features, which is the input file of "feature_matrix_processing.jl" |
| regressor_tuning.jl | Hyperparameters tuning for the base models |
| feature_matrix_F3.csv | The data matrix for model training, which contains target Psum and features |
| model_stack_searching.jl | Searching for the best base model combinations for model stack |
| feature_analysis_single.jl | Feature importance analysis: test features one by one |
| feature_analysis_double.jl | Feature importance analysis: remove 2 features at one time |
| feature_analysis_Luck.jl | Feature importance analysis: dropping "smallest" is prefixed, then a random removal of features in combinations of 4 to 8 features was performed |
| feature_names.csv | This file contains all the feature names except for "smallest". This file will be used by "feature_analysis_Luck.jl" |
| feature_analysis_Luck7.jl | Feature importance analysis: randomly remove features in combinations of 7 features |
| feature_type_assignment.jl | Assign features to feature types |
| all_feature_types.csv | Contains feature names and the corresponding feature types |

If you have any questions about the code in this repository, please contact us at: wang.13377@osu.edu; artsimovitch.1@osu.edu
