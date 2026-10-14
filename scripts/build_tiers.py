"""
Thesis pipeline, steps 2-4:
  2. Assemble & clean Tier 1 (pooled multi-family) and Tier 2 (half-Heusler) datasets
     from the Starrydata2 ThermoelectricMaterials dump.
  3. Compute the SPB/Pisarenko (deformation-potential-scattering) baseline and the
     residual (measured S - baseline S) for every matched (T, n, S) point.
  4. Set the validation protocol:
       - Tier 1: family-wise split (leave-one-family-out folds)
       - Tier 2: leave-one-out CV, sample-level, ready for bootstrap on top (per the
         user's decision: LOO + bootstrap for the small-sample mitigation)

Real data in, real numbers out - nothing here is illustrative/fabricated.
"""
import ast
import json
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = _HERE   # expects raw_starrydata2_curves.csv.gz / raw_starrydata2_samples.csv.gz here
UPLOAD_DIR = _HERE  # expects tier2_input_halfheusler_samples.csv here
OUT_DIR = _HERE

KB = 1.380649e-23      # J/K
E = 1.602176634e-19    # C
H = 6.62607015e-34     # J*s
ME = 9.1093837015e-31  # kg

# ---------------------------------------------------------------------------
# 1. Precompute the SPB (deformation-potential-scattering, r=-1/2) transport
#    functions on an eta grid, once, then interpolate for speed.
#    n(eta) reduced  = F_{1/2}(eta)
#    S(eta)          = (kB/e) [ 2 F1(eta)/F0(eta) - eta ]
# ---------------------------------------------------------------------------

def fermi_dirac(j, eta):
    f = lambda x: (x ** j) / (1.0 + np.exp(np.clip(x - eta, -700, 700)))
    val, _ = quad(f, 0, np.inf, limit=200)
    return val

ETA_GRID = np.linspace(-15, 20, 400)
F0 = np.array([fermi_dirac(0.0, e) for e in ETA_GRID])
F1 = np.array([fermi_dirac(1.0, e) for e in ETA_GRID])
FHALF = np.array([fermi_dirac(0.5, e) for e in ETA_GRID])
S_GRID = (KB / E) * (2.0 * F1 / F0 - ETA_GRID)  # V/K, monotonically decreasing in eta

# n_reduced = F_{1/2}(eta) is monotone increasing -> invert via interpolation
eta_of_lnFhalf = interp1d(np.log(FHALF), ETA_GRID, bounds_error=False,
                           fill_value=(ETA_GRID[0], ETA_GRID[-1]))
S_of_eta = interp1d(ETA_GRID, S_GRID, bounds_error=False,
                     fill_value=(S_GRID[0], S_GRID[-1]))

def spb_seebeck(n_m3, T, mstar_ratio):
    """n in m^-3, T in K, mstar_ratio = m*/m_e. Returns S in V/K (signed like data,
    i.e. this returns the magnitude for n-type convention; sign handled by caller)."""
    mstar = mstar_ratio * ME
    Nc = 4.0 * np.pi * (2.0 * mstar * KB * T / H ** 2) ** 1.5
    n_reduced = n_m3 / Nc
    n_reduced = np.clip(n_reduced, 1e-6, None)
    eta = eta_of_lnFhalf(np.log(n_reduced))
    return S_of_eta(eta)


def fit_mstar(T, n_m3, S_abs):
    """Fit a single m*/m_e for a group of (T, n, |S|) points, least squares."""
    def loss(mstar_ratio):
        pred = spb_seebeck(n_m3, T, mstar_ratio)
        return np.sum((pred - S_abs) ** 2)
    res = minimize_scalar(loss, bounds=(0.02, 25.0), method="bounded")
    return res.x


# ---------------------------------------------------------------------------
# 2. Load samples + curves, parse sample_info, build matched (T, S, n) triples
# ---------------------------------------------------------------------------

def parse_info(info):
    if not isinstance(info, str) or not info.strip():
        return {}
    try:
        return json.loads(info)
    except Exception:
        return {}


def parse_arr(s):
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        return None


print("Loading samples table...")
samples = pd.read_csv(f"{DATA_DIR}/raw_starrydata2_samples.csv.gz")
samples["info"] = samples["sample_info"].apply(parse_info)
samples["family_tag"] = samples["info"].apply(lambda d: (d.get("MaterialFamily") or {}).get("category") or None)
samples["fab_cat"] = samples["info"].apply(lambda d: (d.get("FabricationProcess") or {}).get("category") or None)
samples["fab_comment"] = samples["info"].apply(lambda d: (d.get("FabricationProcess") or {}).get("comment") or None)
samples["density"] = samples["info"].apply(lambda d: (d.get("RelativeDensity") or {}).get("extracted") or None)
samples["grainsize"] = samples["info"].apply(lambda d: (d.get("GrainSize") or {}).get("extracted") or None)
samples = samples.drop(columns=["info"])

print("Loading curves table (this is the big one)...")
curves = pd.read_csv(f"{DATA_DIR}/raw_starrydata2_curves.csv.gz",
                      usecols=["sample_id", "prop_x", "prop_y", "x", "y", "composition"])

seebeck_curves = curves[(curves.prop_x == "Temperature") & (curves.prop_y == "Seebeck coefficient")]
carrier_curves = curves[(curves.prop_x == "Temperature") & (curves.prop_y == "Carrier concentration")]
hall_curves = curves[(curves.prop_x == "Temperature") & (curves.prop_y == "Hall coefficient")]

print(f"Seebeck(T) curves: {len(seebeck_curves)}   Carrier-conc(T) curves: {len(carrier_curves)}   "
      f"Hall(T) curves (as n proxy, n=1/(|R_H| e)): {len(hall_curves)}")

# group carrier curves by sample_id for lookup (a sample can have >1 curve, e.g. multiple figures).
# Hall coefficient(T) curves are converted to a carrier-concentration proxy via the
# single-carrier relation n = 1/(|R_H| * e) (units already m^3/C in this dataset -> n in m^-3)
# and pooled in as an additional source, since direct Carrier-concentration(T) curves alone
# are too sparse to build a usable Tier 1 pool (see the real-yield note below).
carrier_by_sample = {}


def add_points(sid, tx, ny):
    if sid not in carrier_by_sample:
        carrier_by_sample[sid] = ([], [])
    carrier_by_sample[sid][0].append(tx)
    carrier_by_sample[sid][1].append(ny)


for _, row in carrier_curves.iterrows():
    tx, ny = parse_arr(row["x"]), parse_arr(row["y"])
    if tx is None or ny is None or len(tx) < 2:
        continue
    add_points(row["sample_id"], tx, ny)

for _, row in hall_curves.iterrows():
    tx, rh = parse_arr(row["x"]), parse_arr(row["y"])
    if tx is None or rh is None or len(tx) < 2:
        continue
    rh = np.abs(rh)
    rh = np.where(rh > 0, rh, np.nan)
    n_from_hall = 1.0 / (rh * E)  # m^-3
    add_points(row["sample_id"], tx, n_from_hall)

carrier_by_sample = {
    sid: (np.concatenate(ts), np.concatenate(ns))
    for sid, (ts, ns) in carrier_by_sample.items()
}

print(f"Samples with a usable carrier-concentration(T) source (direct + Hall-derived): {len(carrier_by_sample)}")

# ---------------------------------------------------------------------------
# Composition-based family classifier — the MaterialFamily metadata tag is only
# filled in for ~39% of samples (a real, direct instance of the metadata-sparsity
# problem this thesis's literature review documents from the sysTEm paper), so
# family assignment falls back to composition-string heuristics, extending the
# half-Heusler/skutterudite regexes already written for this thesis.
# ---------------------------------------------------------------------------
import re

FAMILY_RULES = [
    ("Half-Heusler", re.compile(r"(Ti|Zr|Hf|V|Nb|Ta|Sc|Y|La|Ce)\w*.*(Ni|Co|Fe|Pd|Pt|Rh)\w*.*(Sn|Sb|Bi|As|Ge)")),
    ("Skutterudite", re.compile(r"(Co|Rh|Ir)\d*\.?\d*Sb\d*\.?\d{2,}")),
    ("Bi2Te3-based", re.compile(r"Bi\d*\.?\d*.*Te\d")),
    ("Sb2Te3/Ge-Sb-Te", re.compile(r"Sb\d*\.?\d*Te\d") ),
    ("PbTe-based", re.compile(r"Pb\d*\.?\d*Te\d")),
    ("SnSe-based", re.compile(r"Sn\d*\.?\d*Se\d")),
    ("SnS-based", re.compile(r"Sn\d*\.?\d*S\d(?!e)")),
    ("SnTe-based", re.compile(r"Sn\d*\.?\d*Te\d")),
    ("GeTe/GeSe", re.compile(r"Ge\d*\.?\d*(Te|Se)\d")),
    ("SiGe alloy", re.compile(r"^(Si\d*\.?\d*Ge\d*\.?\d*|Ge\d*\.?\d*Si\d*\.?\d*)$")),
    ("Silicide (CrSi2/FeSi2 etc.)", re.compile(r"(Cr|Fe|Mn|Mg)\d*\.?\d*Si\d")),
    ("Mg2Si/Sn", re.compile(r"Mg2(Si|Sn)")),
    ("Zintl (Ca/Sr/Eu/Yb-Zn/Al-Sb)", re.compile(r"(Ca|Sr|Eu|Yb)\d*\.?\d*.*(Zn|Al|Cd)\d*\.?\d*Sb")),
    ("Cu-chalcogenide (Cu2Se/CuInTe2 etc.)", re.compile(r"Cu\d*\.?\d*(Se|S|Te|In)")),
    ("In-Te/In-Se chalcogenide", re.compile(r"^In\d*\.?\d*(Te|Se)")),
    ("Ag-chalcogenide", re.compile(r"^Ag\d*\.?\d*(Ga|S|Se|Te)")),
    ("Oxide/Cobaltite", re.compile(r"(Co|Zn)\w*O\d|^Zn?O$|CrO|Pb\d*\.?\d*CrO")),
]


def classify_family(tag, composition):
    if isinstance(tag, str) and tag.strip():
        return tag
    comp = str(composition)
    for name, pat in FAMILY_RULES:
        if pat.search(comp):
            return name
    return None


samples["family_assigned"] = samples.apply(
    lambda r: classify_family(r["family_tag"], r["composition"]), axis=1)


def extract_matched_points(seebeck_row_group, min_T=150, max_T=1200):
    """For one sample_id's Seebeck(T) curve(s), interpolate the matching n(T)
    curve onto the same T grid (only within the n-curve's own T range)."""
    sid = seebeck_row_group.name
    if sid not in carrier_by_sample:
        return None
    Tn, nn = carrier_by_sample[sid]
    order = np.argsort(Tn)
    Tn, nn = Tn[order], nn[order]
    # de-duplicate T for interp1d
    Tn_u, idx = np.unique(Tn, return_index=True)
    nn_u = nn[idx]
    if len(Tn_u) < 2:
        return None
    n_interp = interp1d(Tn_u, nn_u, bounds_error=False, fill_value=np.nan)

    rows = []
    for _, row in seebeck_row_group.iterrows():
        Ts, Ss = parse_arr(row["x"]), parse_arr(row["y"])
        if Ts is None or Ss is None:
            continue
        for T, S in zip(Ts, Ss):
            if not (min_T <= T <= max_T):
                continue
            if T < Tn_u.min() or T > Tn_u.max():
                continue
            n_val = float(n_interp(T))
            if not np.isfinite(n_val) or n_val <= 0:
                continue
            rows.append((sid, T, S, n_val))
    return rows


print("Matching Seebeck and carrier-concentration curves per sample...")
all_rows = []
for sid, grp in seebeck_curves.groupby("sample_id"):
    grp.name = sid
    r = extract_matched_points(grp)
    if r:
        all_rows.extend(r)

matched = pd.DataFrame(all_rows, columns=["sample_id", "T", "S_measured", "n_measured"])
print(f"Matched (T, S, n) points before quality filtering: {len(matched)}")

# unit sanity: |S| should be < ~0.002 V/K (2000 uV/K) for real TE materials;
# n should be within a broad physically sane window for doped semiconductors
matched = matched[(matched["S_measured"].abs() < 0.002) &
                   (matched["n_measured"] > 1e14) & (matched["n_measured"] < 1e23)]
print(f"Matched points after basic physical-range filtering: {len(matched)}")

matched = matched.merge(
    samples[["sample_id", "composition", "family_tag", "family_assigned", "fab_cat", "fab_comment", "density", "grainsize"]],
    on="sample_id", how="left")

# ---------------------------------------------------------------------------
# TIER 1 — pooled multi-family set
# ---------------------------------------------------------------------------
tier1 = matched.dropna(subset=["family_assigned"]).copy()
tier1 = tier1.rename(columns={"family_assigned": "family"})
family_counts = tier1.groupby("family")["sample_id"].nunique().sort_values(ascending=False)
KEEP_FAMILIES = family_counts[family_counts >= 3].index.tolist()
tier1 = tier1[tier1["family"].isin(KEEP_FAMILIES)].copy()

print(f"\nTIER 1: {tier1['sample_id'].nunique()} samples, {tier1['family'].nunique()} families "
      f"(family kept if >=3 distinct samples), {len(tier1)} (T,S,n) points")

# log-n descriptor (carrier concentration in cm^-3 for readability)
tier1["n_cm3"] = tier1["n_measured"] / 1e6
tier1["log10_n"] = np.log10(tier1["n_cm3"])

# fit one m*/m_e per family, compute baseline + residual
tier1_out = []
mstar_by_family = {}
for fam, grp in tier1.groupby("family"):
    T = grp["T"].values
    n_m3 = grp["n_measured"].values
    S_abs = grp["S_measured"].abs().values
    if len(grp) >= 3:
        mstar = fit_mstar(T, n_m3, S_abs)
    else:
        mstar = 1.0
    mstar_by_family[fam] = mstar
    pred = spb_seebeck(n_m3, T, mstar)
    signed_pred = np.sign(grp["S_measured"].values) * pred
    g = grp.copy()
    g["mstar_over_me_family_fit"] = mstar
    g["S_baseline_V_per_K"] = signed_pred
    g["S_residual_V_per_K"] = g["S_measured"] - g["S_baseline_V_per_K"]
    tier1_out.append(g)
tier1_final = pd.concat(tier1_out, ignore_index=True)

resid_uVK = tier1_final["S_residual_V_per_K"] * 1e6
rmse = np.sqrt(np.mean(resid_uVK ** 2))
mae = np.mean(np.abs(resid_uVK))
print(f"TIER 1 SPB baseline fit (per-family m*): RMSE={rmse:.1f} uV/K, MAE={mae:.1f} uV/K "
      f"(measured |S| median = {tier1_final['S_measured'].abs().median()*1e6:.1f} uV/K)")

# family-wise leave-one-family-out fold assignment (for use downstream)
families_sorted = sorted(tier1_final["family"].unique())
tier1_final["loFo_fold"] = tier1_final["family"].apply(lambda f: families_sorted.index(f))

tier1_final.to_csv(f"{OUT_DIR}/tier1_pooled_baseline_residuals.csv", index=False)

# ---------------------------------------------------------------------------
# TIER 2 — half-Heusler
# ---------------------------------------------------------------------------
hh_all = pd.read_csv(f"{UPLOAD_DIR}/tier2_input_halfheusler_samples.csv")
hh_ids = set(hh_all["sample_id"])
tier2 = matched[matched["sample_id"].isin(hh_ids)].copy()
tier2["n_cm3"] = tier2["n_measured"] / 1e6
tier2["log10_n"] = np.log10(tier2["n_cm3"])

n_samples_t2 = tier2["sample_id"].nunique()
n_with_micro = tier2.dropna(subset=["density"])["sample_id"].nunique() + \
               tier2.dropna(subset=["grainsize"])["sample_id"].nunique()
print(f"\nTIER 2 (half-Heusler): {n_samples_t2} samples with matched (T,S,n) points, "
      f"{len(tier2)} points total")

if len(tier2) >= 3:
    mstar_t2 = fit_mstar(tier2["T"].values, tier2["n_measured"].values, tier2["S_measured"].abs().values)
else:
    mstar_t2 = 1.0
pred2 = spb_seebeck(tier2["n_measured"].values, tier2["T"].values, mstar_t2)
tier2["mstar_over_me_fit"] = mstar_t2
tier2["S_baseline_V_per_K"] = np.sign(tier2["S_measured"].values) * pred2
tier2["S_residual_V_per_K"] = tier2["S_measured"] - tier2["S_baseline_V_per_K"]

resid2_uVK = tier2["S_residual_V_per_K"] * 1e6
print(f"TIER 2 SPB baseline fit (single m*={mstar_t2:.2f} m_e): "
      f"RMSE={np.sqrt(np.mean(resid2_uVK**2)):.1f} uV/K, MAE={np.mean(np.abs(resid2_uVK)):.1f} uV/K")

# leave-one-out fold assignment at the SAMPLE level (not point level) so a sample's
# points never span train/test
sample_ids_t2 = sorted(tier2["sample_id"].unique())
tier2["loo_fold"] = tier2["sample_id"].apply(lambda s: sample_ids_t2.index(s))

tier2.to_csv(f"{OUT_DIR}/tier2_halfheusler_baseline_residuals.csv", index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== FAMILY BREAKDOWN (Tier 1) ===")
print(tier1_final.groupby("family")["sample_id"].nunique().sort_values(ascending=False).to_string())

print("\nWrote:")
print(f"  {OUT_DIR}/tier1_pooled_baseline_residuals.csv  ({len(tier1_final)} rows)")
print(f"  {OUT_DIR}/tier2_halfheusler_baseline_residuals.csv  ({len(tier2)} rows)")
