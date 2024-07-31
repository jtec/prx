import numpy as np

from prx import atmospheric_corrections as atmo
from pathlib import Path
import shutil
import pytest
import os
from prx import helpers, converters


@pytest.fixture
def rnx3_input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    rnx3_nav_test_file = test_directory.joinpath("BRDC00IGS_R_20220010000_01D_MN.zip")
    shutil.copy(
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2022001/{rnx3_nav_test_file.name}",
        rnx3_nav_test_file,
    )
    assert rnx3_nav_test_file.exists()

    yield {"rnx3_nav_file": rnx3_nav_test_file}
    shutil.rmtree(test_directory)


def test_get_klobuchar_parameters_from_rinex3(rnx3_input_for_test):
    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        rnx3_input_for_test["rnx3_nav_file"]
    )

    # expected GPSA and GPSB parameters from header
    gps_a_expected = np.array([1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07])
    gps_b_expected = np.array([1.1674e05, -2.2938e05, -1.3107e05, 1.0486e06])

    # Compute RNX3 satellite position
    # load RNX3 NAV file
    nav_ds = helpers.parse_rinex_file(path_to_rnx3_nav_file)

    # recover klobuchar parameters
    gps_a = nav_ds.ionospheric_corr_GPS[0:4]
    gps_b = nav_ds.ionospheric_corr_GPS[4:9]

    assert (gps_a == gps_a_expected).all()
    assert (gps_b == gps_b_expected).all()


def test_klobuchar_correction():
    threshold_iono_error_m = 0.001

    # expected iono correction
    iono_corr_magnitude = np.array(
        [
            [
                3.11827805725116,
                2.76346284622445,
                2.47578416083077,
                2.24496388261401,
                2.06032891260419,
                1.91298307101363,
                1.79609582797929,
                1.70464456302035,
                1.63507066299466,
                1.58498693403951,
                1.55296481004393,
                1.53840074789711,
                1.54145509818181,
            ],
            [
                1.53920028954766,
                1.50821854983715,
                1.49908327097108,
                1.51243673774712,
                1.54997905081108,
                1.61468252478083,
                1.71115564975480,
                1.84619712027765,
                2.02955605453863,
                2.27477164702173,
                2.59937372911026,
                3.02175720678766,
                3.54659048090842,
            ],
            [
                1.87409948444320,
                1.77045425397559,
                1.69072749710603,
                1.63067874821637,
                1.58734346598049,
                1.55875068819254,
                1.54373803510269,
                1.54184164439759,
                1.55324704940871,
                1.57879519961822,
                1.62004630284781,
                1.67941192333763,
                1.76037526636371,
            ],
            [
                3.59539453985740,
                3.21516016338386,
                2.90937389763470,
                2.68006984519335,
                2.52078132484571,
                2.42442804067487,
                2.38622307601646,
                2.40461884807421,
                2.48167682626566,
                2.62331655108814,
                2.83924716365741,
                3.14130628142798,
                3.53626108493281,
            ],
        ]
    )
    # GPSA and GPSB parameters from header of 'datasets/TLSE_2022001/BRDC00IGS_R_2022001000_01D_GN.rnx'
    gps_a = np.array([1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07])
    gps_b = np.array([1.1674e05, -2.2938e05, -1.3107e05, 1.0486e06])

    # define other parameters
    # TLSE coordinates
    lat_u_rad = 0.760277591246057
    lon_u_rad = 0.025846486197371

    tow_s = np.arange(
        519300, 530100 + 1, 900
    )  # added +1 to stop parameter, in order to have the last tow

    el_s_rad = np.array(
        [
            [
                0.389610854319298,
                0.481591693955173,
                0.574237498820712,
                0.667572412962547,
                0.761560089319323,
                0.856042601539353,
                0.950629038638004,
                1.04447640837162,
                1.13581910197266,
                1.22087957072990,
                1.29139789349717,
                1.33117290821177,
                1.32225404957264,
            ],
            [
                1.32880383278317,
                1.45383286061390,
                1.55739366118781,
                1.42982587282686,
                1.29903323460201,
                1.16780305420549,
                1.03698416874356,
                0.907343960585231,
                0.779567544648903,
                0.654243411550507,
                0.531861934393996,
                0.412825197381997,
                0.297466687636666,
            ],
            [
                0.885115729883922,
                0.974746626276506,
                1.06100972686825,
                1.14245728631761,
                1.21631913957279,
                1.27722566221345,
                1.31580808072399,
                1.32114989113608,
                1.29068803221845,
                1.23320643133462,
                1.15908793116013,
                1.07501620545686,
                0.984645953599474,
            ],
            [
                0.287762609990547,
                0.367381507071247,
                0.441427226416507,
                0.506380124255119,
                0.558295989210626,
                0.593228549087166,
                0.607955032773023,
                0.600797552819170,
                0.572114041632790,
                0.524135943426051,
                0.460262721537637,
                0.384235811361961,
                0.299535600509999,
            ],
        ]
    )
    az_s_rad = np.array(
        [
            [
                4.56363365357862,
                4.65467710508134,
                4.75124176292165,
                4.85403134833483,
                4.96431221855663,
                5.08429983949698,
                5.21787834181750,
                5.37200387955870,
                5.55952265868135,
                5.80452476654286,
                6.14821969712950,
                0.343178142603494,
                0.880555341913995,
            ],
            [
                5.56302323945390,
                5.58521362127011,
                3.16194222315421,
                2.66393922952610,
                2.68376384054149,
                2.71680614323896,
                2.75058917643377,
                2.78232011519758,
                2.81091902571040,
                2.83578672401276,
                2.85648084028969,
                2.87259868662011,
                2.88372319717759,
            ],
            [
                4.81923414504988,
                4.95369594617507,
                5.10928919834591,
                5.29609360530352,
                5.53094116969519,
                5.83866338287425,
                6.23904313964220,
                0.416913285448056,
                0.843431991132698,
                1.17680175396222,
                1.42588154064883,
                1.61686494076145,
                1.76960252674726,
            ],
            [
                2.05652442143632,
                1.96522817949171,
                1.86382345284427,
                1.75052651625638,
                1.62489637369112,
                1.48877979725775,
                1.34690015307975,
                1.20635637233483,
                1.07484635285706,
                0.958541344097546,
                0.860891557472562,
                0.782742408016806,
                0.723188958927464,
            ],
        ]
    )

    # compute iono correction from Klobuchar model
    iono_corr = atmo.compute_klobuchar_l1_correction(
        tow_s, gps_a, gps_b, el_s_rad, az_s_rad, lat_u_rad, lon_u_rad
    )

    assert np.max(np.fabs(iono_corr - iono_corr_magnitude)) < threshold_iono_error_m


def test_unb3m_corrections():
    # compare the correction computed by atmospheric_corrections.compute_unb3m_corrections and the results computed
    # by the Matlab version provided by the author of the UNB3M model

    # The tropo delay reported in the text file is rounded to the 3rd decimal
    tol = 1e-3

    tropo_expected = np.genfromtxt(
        helpers.prx_repository_root() / "src/prx/tools/UNB3m_pack/tunb3m_.txt",
        skip_header=3,
    )

    lat_rad = np.deg2rad(tropo_expected[:, 0])
    height_m = tropo_expected[:, 1]
    day_of_year = tropo_expected[:, 2]
    elevation_rad = np.deg2rad(tropo_expected[:, 3])

    (
        tropo_delay_m,
        tropo_zhd_m,
        tropo_hydrostatic_mapping,
        tropo_zwd_m,
        tropo_wet_mapping,
    ) = atmo.compute_unb3m_correction(lat_rad, height_m, day_of_year, elevation_rad)

    assert (np.abs(tropo_delay_m - tropo_expected[:, 8]) < tol).all()
