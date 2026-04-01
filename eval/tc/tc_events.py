# tc_events.py
"""
Event and experiment configuration for TC PDF plotting.
No plotting code here.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TCEvent:
    name: str
    year: str
    month: str
    dates: List[str]  # DD
    analysis: str  # e.g. OPER_O320_0001
    analysis_dates: List[str]  # YYYYMMDD
    expid_enfo_o320: str
    expid_eefo_o96: str
    list_expid_ml: List[str]
    area_north: float
    area_west: float
    area_south: float
    area_east: float

    # Plot config
    regrid_resolution: float = 0.25
    mslp_bin_range: Tuple[float, float, float] = (980, 1021, 1)
    wind_bin_range: Tuple[float, float, float] = (0, 35.01, 1)
    mslp_ylim: Tuple[float, float] = (0, 4)
    wind_ylim: Tuple[float, float] = (0, 2)
    plot_title: str = ""
    reference_expids: Tuple[str, ...] = ()


EVENTS: Dict[str, TCEvent] = {
    "franklin": TCEvent(
        name="franklin",
        year="2023",
        month="08",
        dates=["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
        analysis="OPER_O320_0001",
        analysis_dates=["20230820", "20230826"],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O320_0001",
        expid_eefo_o96="EEFO_O96_0001",
        area_north=40.0,
        area_west=-80.0,
        area_south=10.0,
        area_east=-50.0,
        plot_title="Franklin normed pdfs",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001", "ENFO_O320_ip6y"),
    ),
    "idalia": TCEvent(
        name="idalia",
        year="2023",
        month="08",
        dates=["26", "27", "28", "29", "30"],
        analysis="OPER_O320_0001",
        analysis_dates=[
            "20230826",
        ],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O320_0001",
        expid_eefo_o96="EEFO_O96_0001",
        area_north=40.0,
        area_west=-100.0,
        area_south=10.0,
        area_east=-70.0,
        plot_title="Idalia normed pdfs",
        mslp_bin_range=(980, 1021, 1),
        wind_bin_range=(0, 35.01, 1),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 2),
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001", "ENFO_O320_ip6y"),
    ),
    "hilary": TCEvent(
        name="hilary",
        year="2023",
        month="08",
        dates=["16", "17", "18", "19", "20"],
        analysis="OPER_O320_0001",
        analysis_dates=[
            "20230816",
        ],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O320_0001",
        expid_eefo_o96="EEFO_O96_0001",
        area_north=35.0,
        area_west=-125.0,
        area_south=0.0,
        area_east=-95.0,
        plot_title="Hilary normed pdfs",
        mslp_bin_range=(960, 1021, 1),
        wind_bin_range=(0, 30.01, 1),
        mslp_ylim=(0, 2),
        wind_ylim=(0, 2),
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001", "ENFO_O320_ip6y"),
    ),
    "dora": TCEvent(
        name="dora",
        year="2023",
        month="08",
        dates=[
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
        ],
        analysis="OPER_O320_0001",
        analysis_dates=[
            "20230801",
            "20230807",
            "20230813",
        ],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O320_0001",
        expid_eefo_o96="EEFO_O96_0001",
        area_north=25.0,
        area_west=175.0,
        area_south=5.0,
        area_east=-105.0,
        plot_title="Dora normed pdfs",
        mslp_bin_range=(970, 1021, 1),
        wind_bin_range=(0, 35.01, 1),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 2),
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001", "ENFO_O320_ip6y"),
    ),
    "fernanda": TCEvent(
        name="fernanda",
        year="2023",
        month="08",
        dates=["12", "13", "14", "15", "16", "17"],
        analysis="OPER_O320_0001",
        analysis_dates=[
            "20230812",
        ],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O320_0001",
        expid_eefo_o96="EEFO_O96_0001",
        area_north=30.0,
        area_west=-135.0,
        area_south=0.0,
        area_east=-105.0,
        plot_title="Fernanda normed pdfs",
        mslp_bin_range=(980, 1021, 1),
        wind_bin_range=(0, 30.01, 1),
        mslp_ylim=(0, 10),
        wind_ylim=(0, 5),
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001", "ENFO_O320_ip6y"),
    ),
    "humberto": TCEvent(
        name="humberto",
        year="2025",
        month="09",
        dates=["26", "27", "28", "29", "30"],
        analysis="OPER_O96_0001",
        analysis_dates=[
            "20250926",
        ],
        list_expid_ml=[],
        expid_enfo_o320="ENFO_O48_0001",
        expid_eefo_o96="ENFO_O96_0001",
        area_north=45.0,
        area_west=-90.0,
        area_south=15.0,
        area_east=-50.0,
        plot_title="Humberto normed pdfs",
        mslp_bin_range=(980, 1021, 1),
        wind_bin_range=(0, 35.01, 1),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 2),
        reference_expids=("ENFO_O48_0001",),
    ),
}
