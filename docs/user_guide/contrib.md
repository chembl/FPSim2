# Contributor modules

## [iSIM](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00041b) 

iSIM performs comparisons of multiple molecules at the same time and yields the same value as the average pairwise comparisons of molecules represented by binary fingerprints.

```python
from FPSim2.contrib.isim.isim_comp import get_sim_dict
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_34.h5'
fpe = FPSim2Engine(fp_filename)

# get average similarity of the dataset
sim_dict = get_sim_dict(fpe)

# calculate medoid of the dataset
medoid = calculate_medoid(fpe)

# calculate outlier of the dataset
outlier = calculate_outlier(fpe)

# calculate complementary similarity
comp_sim = calculate_comp_sim(fpe)
```