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
| size.csv, GC.csv, topology.csv, repT-ID.csv, repT.csv, relaxaseT-ID.csv, relaxaseT.csv, mpfT-ID.csv, mpfT.csv, oriT-ID.csv, oriT.csv | original categorical values of these features |
| features_pfamscan.sh | Calculate the percentage of the pfam_models in each cluster |
| pfamscan_CL.txt | It contains the file names that need to be processed by features_pfamscan.sh |
| pfamscan_CL.zip | A total of 1125 files corresponding to the 1125 clusters. The predicted pfam_models are listed in each file | 
