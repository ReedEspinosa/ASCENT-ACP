"""Configuration dataclasses for the ASCENT-ACP processing pipeline.

A single :class:`PipelineConfig` drives the full chain:
merged pickle -> filtering -> window averaging -> ISARA retrieval -> netCDF.
Configs round-trip to JSON (see ``configs/activate_*.json``) and the resolved
JSON is embedded in the output netCDF for provenance.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

_PACKAGE_DATA = Path(__file__).parent / "data"


@dataclass
class PathsConfig:
    input_pkl: str = ""
    meta_pickle: str = ""
    output_dir: str = "."
    isara_code_dir: str = "/Users/wrespino/Synced/Local_Code_MacBook/ISARA_code"
    mopsmap_executable: str = (
        "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/mopsmap"
    )
    optical_dataset_dir: str = (
        "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/optical_dataset/"
    )
    scratch_dir: str = "/tmp/ascent_acp_scratch"  # cwd for MOPSMAP temp files
    lut_cache_dir: str = ""  # optics LUT cache; "" -> <output_dir>/lut_cache
    # Clock-alignment provenance (the /clock_alignment netCDF group). Empty
    # diagnostics path -> derive "shift_diagnostics_<input_pkl_basename>.csv".
    shift_table_csv: str = ""       # variable -> shift_group table
    shift_diagnostics_csv: str = ""  # per (date x shift_group) applied shifts
    family_map_json: str = ""       # "" -> data/<campaign>_instrument_families.json


@dataclass
class ChannelConfig:
    """Column-name suffixes (resolved against the merged DataFrame by varmap)
    and the wavelength layout passed to ISARA."""

    sca_suffixes: dict = field(
        default_factory=lambda: {
            "450": "Sc450_submicron",
            "550": "Sc550_submicron",
            "700": "Sc700_submicron",
        }
    )
    abs_suffixes: dict = field(
        default_factory=lambda: {
            "470": "Abs470_total",
            "532": "Abs532_total",
            "660": "Abs660_total",
        }
    )
    ssa_suffixes: dict = field(
        default_factory=lambda: {
            "450": "SSA_450nm",
            "550": "SSA_550nm",
            "700": "SSA_700nm",
        }
    )
    rh_sc_suffix: str = "RH_Sc_submicron"
    gamma_suffix: str = "gamma550"
    frh_suffix: str = "fRH550_RH20to80"
    ae_suffix: str = "AEscat_450to700nm"
    inlet_flag_suffix: str = "InletFlag_LARGE"
    n_cdp_suffix: str = "N_CDP"
    lwc_cdp_suffix: str = "LWC_CDP"
    n_fcdp_suffix: str = "N_FCDP"
    lwc_fcdp_suffix: str = "LWC_FCDP"
    lat_suffix: str = "Latitude"
    lon_suffix: str = "Longitude"
    alt_suffix: str = "GPS_altitude"
    # Wind pair for vector averaging in the 60 s raw statistics; direction is
    # meteorological (degrees the wind blows FROM, clockwise from north).
    wind_speed_suffix: str = "Wind_Speed"
    wind_dir_suffix: str = "Wind_Direction"
    # Wavelengths (nm) fed to ISARA; keys above must cover these
    dry_wvl_sca: list = field(default_factory=lambda: [450, 550, 700])
    dry_wvl_abs: list = field(default_factory=lambda: [470, 532, 660])
    wet_wvl_sca: list = field(default_factory=lambda: [550])
    val_wvl: list = field(default_factory=list)  # extra output wavelengths


@dataclass
class FilterConfig:
    """Row-level (1 Hz) QC thresholds, after Kacenelenbogen et al. (2022) A1.1."""

    cloud_n_max_cm3: float = 1.0  # CDP/FCDP droplet number above this = cloud
    cloud_lwc_max_gm3: float = 1.0e-3  # LWC above this = cloud
    use_fcdp: bool = True
    cloud_pad_s: int = 5  # dilate cloud mask +/- this many seconds
    require_inlet_flag_zero: bool = True  # keep only isokinetic-inlet samples
    min_dry_sc450_Mm: float = 10.0  # drop rows with dry Sc450 <= this (Mm-1)
    min_ssa: float = 0.7  # drop rows with SSA <= this
    ssa_filter_wvl: str = "550"  # which SSA channel the min_ssa test uses
    dry_ref_rh: float = 40.0  # gamma-correct Sc to this RH when RH_Sc exceeds it
    wet_rh: float = 80.0  # RH of the synthesized humidified scattering


@dataclass
class WindowConfig:
    window_s: int = 60
    min_valid_points: int = 20  # 1 Hz rows required per window
    ae_max_relstd: float = 0.30  # reject window if std(AE)/|mean(AE)| exceeds
    ae_std_mode: str = "relative"  # "relative" or "absolute"
    min_valid_points_per_bin: int = 10  # PSD bin -> NaN if fewer valid samples


@dataclass
class PSDConfig:
    smps_bins_csv: str = str(_PACKAGE_DATA / "ACTIVATE_SMPS_bins.csv")
    las_bins_csv: str = str(_PACKAGE_DATA / "ACTIVATE_LAS_bins.csv")
    variant_name: str = "submicron"  # label recorded in output
    psd_max_um: float = 1.0  # truncate PSD at this diameter (1.0 submicron cut;
    #                          set to inlet_cutoff_um for the total variant)
    inlet_cutoff_um: float = 5.0  # aircraft inlet 50% cutoff; hard upper limit
    smps_min_dp_um: float = 0.0  # optionally trim smallest SMPS bins


@dataclass
class IsaraConfig:
    size_equ: str = "cs"
    shape: str = "sphere"
    nonabs_fraction: float = 0.0
    rho_dry: float = 1.0
    rho_wet: float = 1.0
    num_theta: int = 2
    n_workers: int = 8
    use_lut: bool = True  # precomputed-optics CRI search (falls back per window)
    lut_min_pattern_count: int = 5  # build a LUT only for bin patterns this common


@dataclass
class MergeConfig:
    """Drives the campaign-agnostic ICARTT->pickle merge stage (ASCENT_ACP.merge).

    Generalizes the formerly ACTIVATE-hardcoded wrapper; the merge engine
    (icartt_read_and_merge.icartt_merger) is called unchanged.
    """

    icartt_dir: str = ""            # directory of source .ict files
    instruments: list = field(default_factory=list)  # instrument tokens to include
    # Regex matching the source filenames; must expose named groups 'instr'
    # (validated against `instruments`) and 'date' (parsed with `date_format`).
    filename_regex: str = r"^(?P<instr>[A-Za-z0-9\-]+)_HU25_(?P<date>\d{8})_R\d+(_L\d)?\.ict$"
    date_format: str = "%Y%m%d"
    merge_mode: str = "Merge_Beside"
    master_timeline_step_s: int = 1  # grid spacing (s) of the per-date timeline
    prefix_instr_name: bool = True
    exclude_regexes: list = field(default_factory=list)  # columns to drop post-merge
    n_workers: int = 6
    staging_dir: str = ""           # "" -> /tmp/<campaign>_merge_staging
    # Instrument token whose ICARTT files define the flight envelopes (their
    # first/last data times become takeoff/landing; _L1/_L2 files are separate
    # flights). "" -> fall back to data-presence gap detection on the frame.
    flight_marker_instrument: str = ""


@dataclass
class OutputConfig:
    """Controls the grouped netCDF export (ASCENT_ACP.netcdf_export)."""

    version: str = "V3"             # filename version tag
    emit_observations: bool = True  # /observations native-cadence passthrough
    emit_windowed_raw: bool = True  # /windowed/raw 60 s means of every raw column
    float32: bool = True            # store float vars as float32
    compression_level: int = 4      # zlib complevel (0 = off)


@dataclass
class PipelineConfig:
    campaign: str = "ACTIVATE"
    year: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    psd: PSDConfig = field(default_factory=PSDConfig)
    isara: IsaraConfig = field(default_factory=IsaraConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_json(self, path=None):
        txt = json.dumps(dataclasses.asdict(self), indent=2)
        if path is not None:
            Path(path).write_text(txt)
        return txt

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text())
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw):
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in raw:
                continue
            val = raw[f.name]
            sub = {
                "paths": PathsConfig,
                "channels": ChannelConfig,
                "filters": FilterConfig,
                "window": WindowConfig,
                "psd": PSDConfig,
                "isara": IsaraConfig,
                "merge": MergeConfig,
                "output": OutputConfig,
            }.get(f.name)
            kwargs[f.name] = sub(**val) if sub else val
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**kwargs)
