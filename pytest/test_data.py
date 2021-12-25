import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
import geopandas


def test_get_gis():
    urls = [
        "https://www.eia.gov/maps/map_data/CrudeOil_Pipelines_US_EIA.zip",
        "https://www.eia.gov/maps/map_data/Petroleum_Refineries_US_EIA.zip",
        "https://gis.energy.gov.ab.ca/GeoviewData/OS_Agreements_Shape.zip",
    ]

    for url in urls:
        gf = rt.data.get_gis(url)

        assert isinstance(
            gf, geopandas.GeoDataFrame
        ), f"get_gis failed to return geopandas dataframe from {url}"

