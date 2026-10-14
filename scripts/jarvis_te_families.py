"""
Download JARVIS-DFT (dft_3d) data and split into two thermoelectric families:
  1. Bismuth-telluride / tetradymite family  (Bi2Te3, Sb2Te3, Bi2Se3, and related
     Bi-Sb-Te-Se chalcogenides)
  2. Half-Heusler family (ABC, 1:1:1 ternary compounds, typically space group
     F-43m / #216, the classic half-Heusler structure)

Requires: pip install jarvis-tools pandas --break-system-packages
Needs internet access to figshare.com (this data is NOT hosted on pypi/github,
so run this on your own machine / Colab, not in a network-restricted sandbox).
"""

import pandas as pd
from jarvis.db.figshare import data as jdata

# -------------------------------------------------------------------
# 1. Download the full dft_3d dataset (one-time download, cached after)
# -------------------------------------------------------------------
print("Downloading JARVIS dft_3d dataset (this can take a few minutes)...")
d = jdata("dft_3d")          # use "dft_3d_2021" or "dft_3d_2022" for a fixed snapshot
df = pd.DataFrame(d)
print(f"Total entries in dft_3d: {len(df)}")

# -------------------------------------------------------------------
# Helper: pull out element list from the JARVIS 'atoms' field
# -------------------------------------------------------------------
def get_elements(row):
    try:
        return set(row["atoms"]["elements"])
    except Exception:
        return set()

df["elements"] = df.apply(get_elements, axis=1)
df["n_elements"] = df["elements"].apply(len)

# Keep only entries that actually have computed Seebeck / power factor data
def has_te_data(row):
    s_n = row.get("n-Seebeck", "na")
    s_p = row.get("p-Seebeck", "na")
    return (s_n not in ("na", None)) or (s_p not in ("na", None))

df["has_te_data"] = df.apply(has_te_data, axis=1)

te_cols = [
    "jid", "formula", "spg_number", "spg_symbol", "crys",
    "optb88vdw_bandgap", "mbj_bandgap",
    "n-Seebeck", "p-Seebeck",
    "n-powerfact", "p-powerfact",
    "avg_elec_mass", "avg_hole_mass",
    "formation_energy_peratom", "ehull",
]
te_cols = [c for c in te_cols if c in df.columns]  # keep only cols that exist

# -------------------------------------------------------------------
# 2. Bismuth telluride / tetradymite family
#    -> compounds containing Bi/Sb (pnictogen) AND Te/Se (chalcogen)
# -------------------------------------------------------------------
pnictogens = {"Bi", "Sb"}
chalcogens = {"Te", "Se", "S"}

def is_bite_family(elems):
    return bool(elems & pnictogens) and bool(elems & chalcogens)

bite_mask = df["elements"].apply(is_bite_family) & df["has_te_data"]
df_bite = df.loc[bite_mask, te_cols].reset_index(drop=True)
print(f"Bi2Te3 / tetradymite-family entries with TE data: {len(df_bite)}")

df_bite.to_csv("bismuth_telluride_family_TE.csv", index=False)

# -------------------------------------------------------------------
# 3. Half-Heusler family
#    -> ternary (3-element) compounds with 1:1:1 stoichiometry,
#       space group F-43m (#216) is the standard half-Heusler group
# -------------------------------------------------------------------
def is_half_heusler(row):
    if row["n_elements"] != 3:
        return False
    try:
        spg = int(row.get("spg_number", -1))
    except (ValueError, TypeError):
        return False
    if spg != 216:                      # F-43m
        return False
    # check roughly 1:1:1 stoichiometry from the reduced formula
    try:
        comp = row["atoms"]["elements"]
        counts = {e: comp.count(e) for e in set(comp)}
        vals = list(counts.values())
        return len(vals) == 3 and max(vals) == min(vals)
    except Exception:
        return False

hh_mask = df.apply(is_half_heusler, axis=1) & df["has_te_data"]
df_hh = df.loc[hh_mask, te_cols].reset_index(drop=True)
print(f"Half-Heusler entries with TE data: {len(df_hh)}")

df_hh.to_csv("half_heusler_family_TE.csv", index=False)

print("\nSaved:")
print(" - bismuth_telluride_family_TE.csv")
print(" - half_heusler_family_TE.csv")
