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
