# plasmid_project_IAlab

Folder "data_file" contains the input files for the scripts.

| Files | Description |
| --- | --- |
| leidenalg_CPM.py | Plasmid clustering by using leidenalg with CPM |
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
| feature_matrix_F.csv | The data matrix for model training, which contains target Psum and features |
| model_stack_searching.jl | Searching for the best base model combinations for model stack |
| feature_analysis_round1.jl | Feature importance analysis: test features one by one |
| feature_analysis_rounds.jl | Feature importance analysis: remove 2 features at one time |
| feature_analysis_Luck1.jl | Feature importance analysis: randomly remove features in combinations of 3 to 20 features |
| feature_analysis_Luck2.jl | Feature importance analysis: fix "smallest" to 0 and then randomly remove features in combinations of 16 to 20 features |
| feature_names.csv | Contain all feature names except for "smallest". It will be used by feature_analysis_Luck2.jl |
