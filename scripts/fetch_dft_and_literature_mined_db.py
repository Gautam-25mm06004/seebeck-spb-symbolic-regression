"""
Two datasets I could NOT reach from my sandbox (outbound network policy blocks
figshare.com, api.figshare.com, datadryad.org, and materialsproject.org from
here) but that you should be able to reach normally. Run this on your own
machine, then upload the two output CSVs (or just the raw downloaded files)
back into this chat and I'll do the half-Heusler/skutterudite filtering and
cross-checking for you, same as I already did for Starrydata2.

DATASET 1: Ricci et al. ab initio electronic transport database
  (this is your DFT base-equation data source -- Seebeck coefficient tabulated
  over grids of carrier concentration n and temperature T, computed with
  BoltzTraP from Materials Project band structures)
  Paper: https://www.nature.com/articles/sdata201785
  Data (figshare, tabular form): https://figshare.com/articles/dataset/ricci_boltztrap_mp_tabular/14701110
  Data (Dryad, original release): https://datadryad.org/resource/doi:10.5061/dryad.gn001
  Also queryable live via the Materials Project API (needs a free API key
  from https://materialsproject.org/api) using pymatgen/mp-api, which is
  actually the more robust route since you can query "BoltzTrapDoc" for
  arbitrary structures rather than relying on a static export.

DATASET 2: ChemDataExtractor-mined thermoelectric literature database
  (this is your best lead for the EXPERIMENTAL side WITH a fabrication-process
  field already extracted, separate from Starrydata2 -- worth cross-checking
  against what Starrydata2 gave us)
  Paper: https://www.nature.com/articles/s41597-022-01752-1
  Data (figshare, CSV/JSON/MongoDB): https://doi.org/10.6084/m9.figshare.19658787
  Code: https://github.com/odysie/thermoelectricsdb

--------------------------------------------------------------------------
STEP A -- download (run on a machine that isn't blocked from figshare/dryad)
--------------------------------------------------------------------------
# Dataset 1 (Ricci/BoltzTraP, ~48 MB):
curl -L -o ricci_boltztrap_mp_tabular.zip \
  "https://figshare.com/ndownloader/articles/14701110/versions/1"
unzip ricci_boltztrap_mp_tabular.zip -d ricci_boltztrap

# Dataset 2 (ChemDataExtractor TE DB):
curl -L -o thermoelectricsdb.zip \
  "https://figshare.com/ndownloader/articles/19658787/versions/1"
unzip thermoelectricsdb.zip -d thermoelectricsdb

--------------------------------------------------------------------------
STEP B -- filter each for half-Heusler / skutterudite and inspect coverage
--------------------------------------------------------------------------
Adjust the file names below to whatever the unzip actually produces (figshare
zips sometimes contain a single CSV, sometimes several formats -- list the
folder first: `ls ricci_boltztrap/` and `ls thermoelectricsdb/`).
"""
import pandas as pd
import re

# ---- Half-Heusler / skutterudite composition heuristics -------------------
# Half-Heusler: ABX with A in {Ti,Zr,Hf,V,Nb,Ta,Sc,Y,rare earths}, B in
# {Ni,Co,Fe,Pd,Pt,Rh}, X in {Sn,Sb,Bi,As,Ge}. Skutterudite: CoSb3-type
# (Co/Rh/Ir with Sb in ~1:3 ratio) or filled variants (rare-earth/alkali
# filler + Co/Fe + P/As/Sb in ~1:4:12 ratio).
HH_A = r"(Ti|Zr|Hf|V|Nb|Ta|Sc|Y|La|Ce|Er|Lu|Dy|Ho|Tm|Yb|Gd|Tb)"
HH_B = r"(Ni|Co|Fe|Pd|Pt|Rh|Ru)"
HH_X = r"(Sn|Sb|Bi|As|Ge)"
HH_PATTERN = re.compile(HH_A + r".*" + HH_B + r".*" + HH_X)
SKUTT_PATTERN = re.compile(r"(Co|Rh|Ir|Fe|Ni)\d*\.?\d*(Sb|As|P)\d*\.?\d*")

def filter_family(df, composition_col, family):
    comp = df[composition_col].astype(str)
    if family == "half-heusler":
        mask = comp.apply(lambda s: bool(HH_PATTERN.search(s)))
    elif family == "skutterudite":
        mask = comp.apply(lambda s: bool(SKUTT_PATTERN.search(s)) and "Sb" in s)
    else:
        raise ValueError(family)
    return df[mask]

if __name__ == "__main__":
    # ---- Dataset 1: Ricci/BoltzTraP DFT data -------------------------------
    # df1 = pd.read_csv("ricci_boltztrap/<whatever the csv is called>.csv")
    # print(df1.columns.tolist())
    # hh_dft = filter_family(df1, "formula", "half-heusler")
    # skutt_dft = filter_family(df1, "formula", "skutterudite")
    # print("half-Heusler DFT entries:", len(hh_dft))
    # print("skutterudite DFT entries:", len(skutt_dft))
    # hh_dft.to_csv("ricci_halfheusler.csv", index=False)
    # skutt_dft.to_csv("ricci_skutterudite.csv", index=False)

    # ---- Dataset 2: ChemDataExtractor literature-mined DB ------------------
    # df2 = pd.read_csv("thermoelectricsdb/<whatever the csv is called>.csv")
    # print(df2.columns.tolist())  # look for a 'Process' or similar field
    # hh_lit = filter_family(df2, "chemical_name_or_formula_col", "half-heusler")
    # skutt_lit = filter_family(df2, "chemical_name_or_formula_col", "skutterudite")
    # print(hh_lit["Process"].value_counts())
    # print(skutt_lit["Process"].value_counts())
    # hh_lit.to_csv("cde_halfheusler.csv", index=False)
    # skutt_lit.to_csv("cde_skutterudite.csv", index=False)
    print("Uncomment the relevant block above once you know the actual column names")
    print("from `df.columns.tolist()` for each file -- I couldn't inspect the raw")
    print("files myself since figshare/dryad are blocked from my sandbox.")
