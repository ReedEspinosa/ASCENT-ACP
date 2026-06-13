"""Instrument-family assignment and the campaign-agnostic merge filename match."""

import re

import pytest

from ASCENT_ACP import families
from ASCENT_ACP.config import PipelineConfig

OPT = "In-situ_optical_aerosol_measurements_from_the_NASA_HU-25"
SMPS = "In-situ_Submicron_Particle_Size_Distributions_from_the_SMPS_on_the_NASA_HU-25"
LAS = "In-situ_Particle_Size_Distributions_from_the_TSI_LAS_on_the_NASA_HU-25"


def test_activate_map_loads():
    fm = families.load_family_map("ACTIVATE")
    assert fm is not None
    assert "optical" in fm["family_order"]


def test_every_column_in_exactly_one_family():
    fm = families.load_family_map("ACTIVATE")
    cols = [OPT + "_Sc550_submicron", SMPS + "_SMPS_Bin01", LAS + "_LAS_Bin01",
            OPT + "_AEscat_450to700nm"]
    assigned = families.assign_families(cols, fm)
    assert set(assigned) == set(cols)  # exactly one entry per column
    assert assigned[OPT + "_Sc550_submicron"][0] == "optical"
    assert assigned[SMPS + "_SMPS_Bin01"][0] == "aerosol_size_dist"
    assert assigned[LAS + "_LAS_Bin01"][0] == "aerosol_size_dist"


def test_unknown_column_goes_to_other():
    fm = families.load_family_map("ACTIVATE")
    assigned = families.assign_families(["Some_Unknown_Instrument_var"], fm)
    assert assigned["Some_Unknown_Instrument_var"][0] == "other"


def test_longest_prefix_wins():
    # SMPS and LAS share a leading token; longest-prefix match must disambiguate
    fm = families.load_family_map("ACTIVATE")
    a = families.assign_families([SMPS + "_nSMPS", LAS + "_nLAS"], fm)
    assert a[SMPS + "_nSMPS"][1] == SMPS
    assert a[LAS + "_nLAS"][1] == LAS


def test_family_order_puts_other_last():
    fm = families.load_family_map("ACTIVATE")
    order = families.family_order(fm, ["other", "optical", "ccn"])
    assert order[0] == "optical"
    assert order[-1] == "other"


def test_no_map_groups_by_provided_titles():
    # No family map, but meta-derived titles let columns group by instrument
    a = families.assign_families([OPT + "_x"], None, titles=[OPT])
    fam, title = a[OPT + "_x"]
    assert title == OPT
    assert fam  # sanitized non-empty family name


def test_no_map_no_titles_falls_back_to_token():
    a = families.assign_families([OPT + "_x"], None)
    assert a[OPT + "_x"][0] == "other"


def test_merge_filename_regex_matches_activate():
    cfg = PipelineConfig()
    rx = re.compile(cfg.merge.filename_regex)
    m = rx.match("ACTIVATE-LARGE-OPTICAL_HU25_20210513_R0.ict")
    assert m and m.group("instr") == "ACTIVATE-LARGE-OPTICAL"
    assert m.group("date") == "20210513"
    # the 20 Hz DLH variant is a distinct instr token (filtered by membership)
    m2 = rx.match("ACTIVATE-DLH-H2O-20Hz_HU25_20210513_R1.ict")
    assert m2.group("instr") == "ACTIVATE-DLH-H2O-20Hz"
    assert rx.match("notes_readme.txt") is None
