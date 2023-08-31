#!/usr/bin/bash
# calculate the percentage of pfam_models in each cluster  

# "pfamscan_CL.txt" contains all the file names that need to be processed.
for file in $(cat pfamscan_CL.txt)
do

awk '
  {
    count[$2]++;
    total++;
  }
  END {
    for (str in count) {
      percentage = (100 * count[str] / total);
      printf "%s %.2f\n", str, percentage;
    }
  }
' "pfamscan_CL/$file" > "pfamscan_CL_percentage/$file.out"

done
