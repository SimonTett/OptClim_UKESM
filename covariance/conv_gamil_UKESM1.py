# Convert GAMIL3 precip covariances values from m/sec to mm/sec.
# Also add anything missing in error covariance with values set to zero. That is OK for a nudged run!

import numpy as np
import pandas as pd
import pathlib
ref_cov = pd.read_csv(pathlib.Path("obserr2011.csv"),index_col=0)

gamil_file = pathlib.Path("GAMIL3_intvar_2011_nudged_3regions.csv")
outfile = gamil_file.parent / (gamil_file.stem + "_conv.csv")
cov = pd.read_csv(gamil_file, index_col=0)
cols = [c for c in cov.columns if c.startswith('Lprecip')]
scale = pd.Series([1.] * len(cov.columns), index=cov.columns)
scale_cols = cov.columns.str.startswith("Lprecip")
print("Following cols are being scaled by 1000",cov.columns[scale_cols])
scale[scale_cols] = 1000.  # convert from m/sec to kg/m^2/sec
scale_mat = pd.DataFrame(np.outer(scale, scale), index=cov.columns, columns=cov.columns)
convert_cov = cov * scale_mat  # scaled the covariance
# some tests!
# Expect 2*(3*n_rows)-3*3 1000 values
r = convert_cov / cov
expect_1000 = (len(cov.columns) - 3) * 3 + 3 * (len(cov.columns) - 3)
count_1000 = np.isclose(r, 1000.).sum()
if int(count_1000) != expect_1000:
    raise ValueError("Not got expected number of 1000's!")
expect_1e6 = 3 * 3  # products
count_1e6 = np.isclose(r, 1e6).sum()
if int(count_1e6) != expect_1e6:
    raise ValueError("Not got expected number of 1e6's!")
# add on the missing columns (if any)
cols_to_add = set(ref_cov.columns) - set(convert_cov.columns)
print("Following cols are missing and will be set to 0.0",cols_to_add)
for col in cols_to_add: # add in missing columns with value 0
    convert_cov.loc[col,:]=0.0
    convert_cov.loc[:,col] =0.0

print(f"Writing converted cov to {outfile}")
convert_cov.to_csv(outfile)
