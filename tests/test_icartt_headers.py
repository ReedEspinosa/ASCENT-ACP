"""ICARTT header scanning: variables, STP basis, bin tables, flight spans."""

import numpy as np
import pytest

from ASCENT_ACP import icartt_headers as ih


def _write_ict(path, varlines, comments=(), title="My Great Instrument",
               data=("100.0, 1.0",)):
    head = ["PI", "ORG", title, "MISSION", "1, 1",
            "2021, 05, 13, 2023, 01, 01", "1", "Time_Start, sec",
            str(len(varlines)), ", ".join(["1"] * len(varlines)),
            ", ".join(["-9999"] * len(varlines))]
    lines = head + list(varlines) + list(comments)
    n_header = len(lines) + 1
    path.write_text("\n".join([f"{n_header}, 1001"] + lines + list(data)))
    return path


def test_parse_header_variables_and_title(tmp_path):
    p = _write_ict(tmp_path / "a.ict",
                   ["Sc550, Mm-1, AerOpt_Scattering_InSitu_Green_RHsp_PM1_STP, scat at 550",
                    "N_X, cm-3, CldMicro_NumConc_InSitu_Optical_Drop_AMB, droplets"],
                   ["DATA_INFO: reported at standard temperature and pressure"])
    h = ih.parse_header(p)
    assert h.title_clean == "My_Great_Instrument"
    assert len(h.variables) == 2
    v = h.var("Sc550")
    assert v.units == "Mm-1" and "STP" in v.standard
    assert "standard temperature" in h.data_info


def test_measurement_conditions_rules():
    mk = lambda name="x", units="", std="", desc="": ih.VarInfo(name, units, std, desc)
    assert ih.measurement_conditions(mk(std="AerOpt_Sc_STP")) == "STP"
    assert ih.measurement_conditions(mk(std="CldMicro_NumConc_AMB")) == "ambient"
    # variable-name _amb refers to ambient RH; the standard token wins
    assert ih.measurement_conditions(
        mk(name="Sc550_amb", std="AerOpt_Scattering_RHa_PM1_STP")) == "STP"
    assert ih.measurement_conditions(mk(units="ppbv")) == "not_applicable"
    assert ih.measurement_conditions(
        mk(), "Data reported at standard temperature and pressure") == "STP"
    assert ih.measurement_conditions(
        mk(), "at ambient temperature and pressure") == "ambient"
    assert ih.measurement_conditions(mk()) == "unspecified"
    assert ih.measurement_conditions(None) == "unspecified"


def test_bin_table_from_descriptions(tmp_path):
    lines = [f"CDP_Bin{i:02d}, #/cm3, Cld_AMB, dNdlogD_at_bin_center_{c}um_by_CDP"
             for i, c in enumerate([2.5, 3.5, 4.5])]
    h = ih.parse_header(_write_ict(tmp_path / "cdp.ict", lines))
    bt = ih.bin_table(h)
    assert list(bt.center_um) == [2.5, 3.5, 4.5]
    assert bt.columns == ["CDP_Bin00", "CDP_Bin01", "CDP_Bin02"]
    # derived edges are midpoints
    assert bt.lower_um[1] == pytest.approx(3.0)
    assert bt.upper_um[1] == pytest.approx(4.0)


def test_bin_table_from_edge_lists(tmp_path):
    lines = [f"dNdlogD_{i:03d}_FCDP, #/m^3, Cld_AMB, per bin" for i in (3, 4, 5)]
    comments = ["OTHER_COMMENTS: Bin Lower Edges (in um) = [3.0, 4.5, 6.0]",
                " Bin Upper Edges (in um) = [4.5, 6.0, 8.0]"]
    h = ih.parse_header(_write_ict(tmp_path / "fcdp.ict", lines, comments))
    bt = ih.bin_table(h)
    assert np.allclose(bt.lower_um, [3.0, 4.5, 6.0])
    assert np.allclose(bt.upper_um, [4.5, 6.0, 8.0])
    assert np.allclose(bt.center_um, np.sqrt(bt.lower_um * bt.upper_um))


def test_bin_table_from_nm_bounds(tmp_path):
    lines = [f"SMPS_Bin{i:02d}, #/cm3, Aer_STP, dNdlogD" for i in (1, 2)]
    comments = ["OTHER_COMMENTS: Bin parameters (in nm) are:",
                " Lower Bounds: 2.97, 3.36", " Mid points: 3.16, 3.55",
                " Upper Bounds: 3.36, 3.76"]
    h = ih.parse_header(_write_ict(tmp_path / "smps.ict", lines, comments))
    bt = ih.bin_table(h)
    assert np.allclose(bt.center_um, [0.00316, 0.00355])


def test_flight_spans(tmp_path):
    _write_ict(tmp_path / "T-SUM_HU25_20210513_R0.ict", ["X, u, s, d"],
               data=["43200.0, 1", "43201.0, 1", "43500.0, 1"])
    _write_ict(tmp_path / "T-SUM_HU25_20210514_R0_L2.ict", ["X, u, s, d"],
               data=["50000.0, 1", "50100.0, 1"])
    rx = r"^(?P<instr>[A-Za-z0-9\-]+)_HU25_(?P<date>\d{8})_R\d+(_L\d)?\.ict$"
    spans = ih.flight_spans(tmp_path, rx, "T-SUM")
    assert [s.flight_id for s in spans] == ["20210513", "20210514_L2"]
    assert spans[0].start_epoch_s % 86400 == 43200
    assert spans[0].end_epoch_s - spans[0].start_epoch_s == 300
    assert spans[1].leg == 2
