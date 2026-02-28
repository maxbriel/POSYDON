"""Unit tests of posydon/binary_evol/singlestar.py
"""

__authors__ = [
    "Max Briel <max.briel@unige.ch>",
]

# import the module which will be tested
import posydon.binary_evol.singlestar as totest

# aliases
np = totest.np
pd = totest.pd

from inspect import isclass, isfunction
from unittest.mock import MagicMock

from pytest import approx, fixture, raises, warns

from posydon.utils.posydonwarning import Pwarn

# fixtures

@fixture
def default_star():
    """A default SingleStar with no arguments."""
    return totest.SingleStar()


@fixture
def hms_star():
    """An H-rich core H burning star at solar metallicity."""
    return totest.SingleStar(
        state='H-rich_Core_H_burning',
        metallicity=1.0,
        mass=10.0,
    )


@fixture
def he_star():
    """A stripped He core He burning star."""
    return totest.SingleStar(
        state='stripped_He_Core_He_burning',
        metallicity=1.0,
        mass=5.0,
    )


@fixture
def bh_star():
    """A BH compact object star."""
    return totest.SingleStar(
        state='BH',
        metallicity=1.0,
        mass=30.0,
    )


@fixture
def ns_star():
    """A NS compact object star."""
    return totest.SingleStar(
        state='NS',
        metallicity=1.0,
        mass=1.4,
    )


@fixture
def massless_star():
    """A massless remnant star."""
    return totest.SingleStar(
        state='massless_remnant',
        metallicity=1.0,
        mass=0.0,
    )


@fixture
def mock_single_star_run():
    """A mock run object mimicking a single-star PSyGrid run."""
    # columns needed by from_run: star_age + mapped STARPROPERTIES names
    h_cols = [
        'star_age', 'star_mass', 'log_R', 'log_L', 'lg_mdot',
        'lg_system_mdot', 'lg_wind_mdot',
        'he_core_mass', 'he_core_radius', 'c_core_mass', 'c_core_radius',
        'o_core_mass', 'o_core_radius', 'co_core_mass', 'co_core_radius',
        'center_h1', 'center_he4', 'center_c12', 'center_n14', 'center_o16',
        'surface_h1', 'surface_he4', 'surface_c12', 'surface_n14',
        'surface_o16',
        'log_LH', 'log_LHe', 'log_LZ', 'log_Lnuc', 'c12_c12',
        'center_gamma', 'avg_c_in_c_core', 'surf_avg_omega',
        'surf_avg_omega_div_omega_crit', 'total_moment_of_inertia',
        'log_total_angular_momentum', 'spin_parameter',
        'conv_env_top_mass', 'conv_env_bot_mass',
        'conv_env_top_radius', 'conv_env_bot_radius',
        'conv_env_turnover_time_g', 'conv_env_turnover_time_l_b',
        'conv_env_turnover_time_l_t', 'envelope_binding_energy',
        'mass_conv_reg_fortides', 'thickness_conv_reg_fortides',
        'radius_conv_reg_fortides',
        'lambda_CE_1cent', 'lambda_CE_10cent', 'lambda_CE_30cent',
        'lambda_CE_pure_He_star_10cent',
        'total_mass_h1', 'total_mass_he4',
    ]
    dt = [(c, '<f8') for c in h_cols]
    n_steps = 3
    history1 = np.zeros(n_steps, dtype=dt)
    # fill with simple evolving values
    history1['star_age'] = [1e6, 5e6, 1e7]
    history1['star_mass'] = [10.0, 9.8, 9.5]
    history1['log_R'] = [0.5, 0.8, 1.0]
    history1['log_L'] = [3.5, 3.6, 3.7]
    history1['center_h1'] = [0.7, 0.4, 0.01]
    history1['center_he4'] = [0.27, 0.55, 0.95]
    history1['center_c12'] = [0.0, 0.0, 0.001]
    history1['surface_h1'] = [0.7, 0.7, 0.7]
    history1['log_LH'] = [3.5, 3.5, 2.0]
    history1['log_LHe'] = [-10.0, -5.0, 3.0]
    history1['log_Lnuc'] = [3.5, 3.5, 3.0]
    history1['he_core_mass'] = [0.0, 1.0, 3.0]

    # final_values: S1_ prefixed columns
    fv_cols = [('S1_' + c, '<f8') for c in h_cols if c != 'star_age']
    fv_cols += [
        ('S1_avg_c_in_c_core_at_He_depletion', '<f8'),
        ('S1_co_core_mass_at_He_depletion', '<f8'),
    ]
    final_values = np.zeros(1, dtype=fv_cols)[0]
    for c in h_cols:
        if c == 'star_age':
            continue
        final_values['S1_' + c] = history1[c][-1]
    final_values['S1_avg_c_in_c_core_at_He_depletion'] = 0.35
    final_values['S1_co_core_mass_at_He_depletion'] = 1.5

    # initial_values with Z
    initial_values = np.array([(0.0142,)], dtype=[('Z', '<f8')])[0]

    # profile data
    profile_cols = [('mass', '<f8'), ('radius', '<f8')]
    final_profile1 = np.array([(1.0, 0.5), (0.5, 0.3)], dtype=profile_cols)

    run = MagicMock()
    run.history1 = history1
    run.final_values = final_values
    run.initial_values = initial_values
    run.final_profile1 = final_profile1
    return run


# define test classes collecting several test functions

class TestElements:
    """Check for objects which should be elements of the tested module."""

    def test_dir(self):
        elements = {
            'CO_radius', 'EXTRA_STAR_COLUMNS_DTYPES', 'Pwarn',
            'SCALAR_NAMES_DTYPES', 'SN_MODELS', 'STARPROPERTIES',
            'STARPROPERTIES_DTYPES',
            'STAR_ATTRIBUTES_FROM_STAR_HISTORY_SINGLE',
            'SingleStar', 'THRESHOLD_CENTRAL_ABUNDANCE', 'Zsun',
            '__authors__', '__builtins__', '__cached__', '__doc__',
            '__file__', '__loader__', '__name__', '__package__', '__spec__',
            'check_state_of_star', 'convert_star_to_massless_remnant',
            'infer_star_state', 'np', 'pd', 'properties_massless_remnant',
            'zams_table',
        }
        totest_elements = set(dir(totest))
        missing_in_test = elements - totest_elements
        assert len(missing_in_test) == 0, \
            "There are missing objects in " \
            + f"{totest.__name__}: " \
            + f"{missing_in_test}. Please " \
            + "check, whether they have been " \
            + "removed on purpose and update " \
            + "this unit test."
        new_in_test = totest_elements - elements
        assert len(new_in_test) == 0, \
            "There are new objects in " \
            + f"{totest.__name__}: {new_in_test}. " \
            + "Please check, whether they have been " \
            + "added on purpose and update this " \
            + "unit test."

    def test_instance_STARPROPERTIES(self):
        assert isinstance(totest.STARPROPERTIES, list)

    def test_instance_STAR_ATTRIBUTES_FROM_STAR_HISTORY_SINGLE(self):
        assert isinstance(totest.STAR_ATTRIBUTES_FROM_STAR_HISTORY_SINGLE,
                          dict)

    def test_instance_SingleStar(self):
        assert isclass(totest.SingleStar)

    def test_instance_properties_massless_remnant(self):
        assert isfunction(totest.properties_massless_remnant)

    def test_instance_convert_star_to_massless_remnant(self):
        assert isfunction(totest.convert_star_to_massless_remnant)


class TestSTARPROPERTIES:
    """Tests for the STARPROPERTIES list."""

    def test_contains_state(self):
        assert 'state' in totest.STARPROPERTIES

    def test_contains_mass(self):
        assert 'mass' in totest.STARPROPERTIES

    def test_contains_metallicity(self):
        assert 'metallicity' in totest.STARPROPERTIES

    def test_contains_log_R(self):
        assert 'log_R' in totest.STARPROPERTIES

    def test_contains_center_h1(self):
        assert 'center_h1' in totest.STARPROPERTIES

    def test_contains_center_he4(self):
        assert 'center_he4' in totest.STARPROPERTIES

    def test_contains_he_core_mass(self):
        assert 'he_core_mass' in totest.STARPROPERTIES

    def test_contains_spin(self):
        assert 'spin' in totest.STARPROPERTIES

    def test_contains_profile(self):
        assert 'profile' in totest.STARPROPERTIES

    def test_no_duplicates(self):
        assert len(totest.STARPROPERTIES) == len(set(totest.STARPROPERTIES))

    def test_all_strings(self):
        for item in totest.STARPROPERTIES:
            assert isinstance(item, str), f"{item} is not a string"


class TestPropertiesMasslessRemnant:
    """Tests for properties_massless_remnant function."""

    def test_returns_dict(self):
        result = totest.properties_massless_remnant()
        assert isinstance(result, dict)

    def test_state_is_massless_remnant(self):
        result = totest.properties_massless_remnant()
        assert result['state'] == 'massless_remnant'

    def test_mass_is_zero(self):
        result = totest.properties_massless_remnant()
        assert result['mass'] == 0.0

    def test_contains_all_starproperties(self):
        result = totest.properties_massless_remnant()
        for key in totest.STARPROPERTIES:
            assert key in result, f"Missing key: {key}"

    def test_nan_values(self):
        result = totest.properties_massless_remnant()
        for key in totest.STARPROPERTIES:
            if key not in ('state', 'mass'):
                assert np.isnan(result[key]) if isinstance(
                    result[key], float) else True


class TestConvertStarToMasslessRemnant:
    """Tests for convert_star_to_massless_remnant function."""

    def test_converts_state(self, hms_star):
        totest.convert_star_to_massless_remnant(hms_star)
        assert hms_star.state == 'massless_remnant'

    def test_converts_mass(self, hms_star):
        totest.convert_star_to_massless_remnant(hms_star)
        assert hms_star.mass == 0.0

    def test_returns_star(self, hms_star):
        result = totest.convert_star_to_massless_remnant(hms_star)
        assert result is hms_star


class TestSingleStarInit:
    """Tests for SingleStar.__init__ method."""

    def test_default_init(self, default_star):
        assert isinstance(default_star, totest.SingleStar)

    def test_default_metallicity(self, default_star):
        assert default_star.metallicity == 1.0

    def test_default_state(self, default_star):
        # state is not set via kwargs, so the STARPROPERTIES loop assigns
        # the default for 'string' dtype, which is ''
        assert default_star.state == ''

    def test_hms_init(self, hms_star):
        assert hms_star.state == 'H-rich_Core_H_burning'
        assert hms_star.mass == 10.0
        assert hms_star.metallicity == 1.0

    def test_hms_center_h1(self, hms_star):
        # For solar metallicity ZAMS: X = 1 - Y - Z
        # Y = zams_table[1.0] = 0.2703, Z = 1.0*0.0142 = 0.0142
        # X = 1 - 0.2703 - 0.0142 = 0.7155
        expected_X = 1.0 - totest.zams_table[1.0] - 1.0 * totest.Zsun
        assert hms_star.center_h1 == approx(expected_X, abs=1e-10)

    def test_hms_center_he4(self, hms_star):
        # Y from zams_table
        expected_Y = totest.zams_table[1.0]
        assert hms_star.center_he4 == approx(expected_Y, abs=1e-10)

    def test_hms_log_R(self, hms_star):
        # default for Core_H_burning: LOW_LOGR_GUESS = 0.0
        assert hms_star.log_R == 0.0

    def test_hms_he_core_mass(self, hms_star):
        # default for Core_H_burning: 0.0
        assert hms_star.he_core_mass == 0.0

    def test_co_state_bh(self, bh_star):
        # BH with 30 Msun should be inferred as BH
        assert bh_star.state == 'BH'

    def test_co_state_ns(self, ns_star):
        assert ns_star.state == 'NS'

    def test_massless_remnant_init(self, massless_star):
        assert massless_star.state == 'massless_remnant'
        assert massless_star.mass == 0.0

    def test_he_core_he_burning(self, he_star):
        assert he_star.state == 'stripped_He_Core_He_burning'
        # For stripped He star in Core_He_burning:
        # default_log_R = LOW_LOGR_GUESS = 0.0
        assert he_star.log_R == 0.0
        # he_core_mass defaults to mass for Core_He_burning stripped_He
        assert he_star.he_core_mass == 5.0

    def test_shell_burning_state(self):
        star = totest.SingleStar(
            state='H-rich_Shell_H_burning',
            metallicity=1.0,
            mass=5.0,
        )
        # Post-MS: center_h1 = THRESHOLD_CENTRAL_ABUNDANCE = 0.01
        assert star.center_h1 == approx(
            totest.THRESHOLD_CENTRAL_ABUNDANCE, abs=1e-10)

    def test_core_c_burning_state(self):
        star = totest.SingleStar(
            state='H-rich_Core_C_burning',
            metallicity=1.0,
            mass=20.0,
        )
        # Core_C_burning: center_h1 = LOW_ABUNDANCE = 1e-6
        assert star.center_h1 == approx(1e-6, abs=1e-10)
        # center_he4 = LOW_ABUNDANCE = 1e-6
        assert star.center_he4 == approx(1e-6, abs=1e-10)
        # he_core_mass defaults to mass
        assert star.he_core_mass == 20.0

    def test_core_depleted_state(self):
        star = totest.SingleStar(
            state='H-rich_Core_He_depleted',
            metallicity=1.0,
            mass=15.0,
        )
        # Core_depleted: center_h1 = LOW_ABUNDANCE, center_he4 = LOW_ABUNDANCE
        assert star.center_h1 == approx(1e-6, abs=1e-10)
        assert star.center_he4 == approx(1e-6, abs=1e-10)

    def test_invalid_metallicity_raises(self):
        with raises(KeyError):
            totest.SingleStar(
                state='H-rich_Core_H_burning',
                metallicity=999.0,
                mass=10.0,
            )

    def test_natal_kick_array_unpacking(self):
        star = totest.SingleStar(
            state='NS',
            mass=1.4,
            natal_kick_array=[100.0, 1.5, 0.8, 3.0],
        )
        assert star.natal_kick_velocity == 100.0
        assert star.natal_kick_azimuthal_angle == 1.5
        assert star.natal_kick_polar_angle == 0.8
        assert star.natal_kick_mean_anomaly == 3.0

    def test_history_initialized(self, hms_star):
        for prop in totest.STARPROPERTIES:
            hist = getattr(hms_star, prop + '_history')
            assert isinstance(hist, list), \
                f"{prop}_history is not a list"
            assert len(hist) == 1, \
                f"{prop}_history should have length 1"

    def test_extra_kwargs_set(self):
        star = totest.SingleStar(
            state='H-rich_Core_H_burning',
            metallicity=1.0,
            mass=10.0,
            custom_attr=42.0,
        )
        assert star.custom_attr == 42.0

    def test_default_natal_kick_attributes(self, default_star):
        assert default_star.natal_kick_velocity is None
        assert default_star.natal_kick_azimuthal_angle is None
        assert default_star.natal_kick_polar_angle is None
        assert default_star.natal_kick_mean_anomaly is None

    def test_default_sn_attributes(self, default_star):
        assert default_star.SN_type is None
        assert default_star.f_fb is None

    def test_co_state_nan_abundances(self, bh_star):
        assert np.isnan(bh_star.center_h1)
        assert np.isnan(bh_star.center_he4)

    def test_ce_quantities_initialized(self, default_star):
        for quantity in ['m_core_CE', 'r_core_CE']:
            for val in [1, 10, 30, 'pure_He_star_10']:
                attr = f'{quantity}_{val}cent'
                assert hasattr(default_star, attr), \
                    f"Missing CE attribute: {attr}"

    def test_sn_model_attributes_initialized(self, default_star):
        from posydon.grids.SN_MODELS import SN_MODELS
        for model_name in SN_MODELS.keys():
            assert hasattr(default_star, model_name), \
                f"Missing SN model attribute: {model_name}"


class TestSingleStarAppendState:
    """Tests for SingleStar.append_state method."""

    def test_appends_history(self, hms_star):
        hms_star.mass = 9.5
        hms_star.append_state()
        assert len(hms_star.mass_history) == 2
        assert hms_star.mass_history[0] == 10.0
        assert hms_star.mass_history[1] == 9.5

    def test_appends_all_properties(self, hms_star):
        hms_star.append_state()
        for prop in totest.STARPROPERTIES:
            hist = getattr(hms_star, prop + '_history')
            assert len(hist) == 2, \
                f"{prop}_history length should be 2 after append"

    def test_multiple_appends(self, hms_star):
        for i in range(5):
            hms_star.mass = 10.0 - i
            hms_star.append_state()
        assert len(hms_star.mass_history) == 6
        assert hms_star.mass_history[-1] == 6.0


class TestSingleStarRestore:
    """Tests for SingleStar.restore method."""

    def test_restore_to_initial(self, hms_star):
        initial_mass = hms_star.mass
        hms_star.mass = 8.0
        hms_star.append_state()
        hms_star.mass = 6.0
        hms_star.append_state()
        hms_star.restore(i=0)
        assert hms_star.mass == initial_mass
        assert len(hms_star.mass_history) == 1

    def test_restore_to_middle(self, hms_star):
        hms_star.mass = 8.0
        hms_star.append_state()
        hms_star.mass = 6.0
        hms_star.append_state()
        hms_star.restore(i=1)
        assert hms_star.mass == 8.0
        assert len(hms_star.mass_history) == 2

    def test_restore_truncates_history(self, hms_star):
        for i in range(5):
            hms_star.mass = 10.0 - i
            hms_star.append_state()
        hms_star.restore(i=2)
        for prop in totest.STARPROPERTIES:
            hist = getattr(hms_star, prop + '_history')
            assert len(hist) == 3, \
                f"{prop}_history should have length 3 after restore(i=2)"


class TestSingleStarToDf:
    """Tests for SingleStar.to_df method."""

    def test_returns_dataframe(self, hms_star):
        df = hms_star.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_column_count(self, hms_star):
        df = hms_star.to_df()
        # profile is excluded by default
        expected_cols = len(totest.STARPROPERTIES) - 1
        assert len(df.columns) == expected_cols

    def test_single_row(self, hms_star):
        df = hms_star.to_df()
        assert len(df) == 1

    def test_multiple_rows_after_append(self, hms_star):
        hms_star.mass = 9.0
        hms_star.append_state()
        df = hms_star.to_df()
        assert len(df) == 2

    def test_mass_column_value(self, hms_star):
        df = hms_star.to_df()
        assert df['mass'].iloc[0] == approx(10.0)

    def test_prefix(self, hms_star):
        df = hms_star.to_df(prefix='S1_')
        assert 'S1_mass' in df.columns
        assert 'mass' not in df.columns

    def test_ignore_columns(self, hms_star):
        df = hms_star.to_df(ignore_columns=['mass'])
        assert 'mass' not in df.columns

    def test_only_select_columns(self, hms_star):
        df = hms_star.to_df(only_select_columns=['mass', 'state'])
        assert 'mass' in df.columns
        assert 'state' in df.columns
        assert len(df.columns) == 2

    def test_include_profile_false(self, hms_star):
        df = hms_star.to_df()
        assert 'profile' not in df.columns

    def test_include_profile_true(self, hms_star):
        df = hms_star.to_df(include_profile=True)
        assert 'profile' in df.columns

    def test_extra_columns(self, hms_star):
        hms_star.custom_col_history = [42.0]
        df = hms_star.to_df(
            extra_columns={'custom_col_history': 'float64'})
        assert 'custom_col' in df.columns


class TestSingleStarToOnelineDf:
    """Tests for SingleStar.to_oneline_df method."""

    def test_returns_dataframe(self, hms_star):
        df = hms_star.to_oneline_df()
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self, hms_star):
        df = hms_star.to_oneline_df()
        assert len(df) == 1

    def test_initial_final_columns(self, hms_star):
        hms_star.mass = 9.0
        hms_star.append_state()
        df = hms_star.to_oneline_df()
        assert 'mass_i' in df.columns
        assert 'mass_f' in df.columns

    def test_initial_final_values(self, hms_star):
        hms_star.mass = 9.0
        hms_star.append_state()
        df = hms_star.to_oneline_df()
        assert df['mass_i'].iloc[0] == approx(10.0)
        assert df['mass_f'].iloc[0] == approx(9.0)

    def test_prefix(self, hms_star):
        df = hms_star.to_oneline_df(prefix='S1_')
        assert any(col.startswith('S1_') for col in df.columns)

    def test_no_history(self, hms_star):
        df = hms_star.to_oneline_df(history=False)
        assert len(df.columns) == 0

    def test_scalar_names(self, hms_star):
        hms_star.SN_type = 'CCSN'
        df = hms_star.to_oneline_df(scalar_names=['SN_type'])
        assert 'SN_type' in df.columns
        assert df['SN_type'].iloc[0] == 'CCSN'


class TestSingleStarRepr:
    """Tests for SingleStar.__repr__ method."""

    def test_repr_contains_class_name(self, hms_star):
        r = repr(hms_star)
        assert 'SingleStar' in r

    def test_repr_contains_properties(self, hms_star):
        r = repr(hms_star)
        assert 'mass: 10.0' in r
        assert 'state: H-rich_Core_H_burning' in r

    def test_repr_is_string(self, hms_star):
        assert isinstance(repr(hms_star), str)


class TestSingleStarInitExtraCoverage:
    """Additional init tests for remaining branches."""

    def test_stripped_he_shell_burning(self):
        # Covers lines 208-210: stripped_He in shell burning state
        star = totest.SingleStar(
            state='stripped_He_Shell_He_burning',
            metallicity=1.0,
            mass=3.0,
        )
        # stripped He shell burning: center_h1 = LOW_ABUNDANCE
        assert star.center_h1 == approx(1e-6, abs=1e-10)
        # center_he4 = LOW_ABUNDANCE
        assert star.center_he4 == approx(1e-6, abs=1e-10)
        # he_core_mass = mass for stripped_He
        assert star.he_core_mass == 3.0

    def test_co_state_inferred_differs(self):
        # Covers line 257: CO state where inferred != given
        # A 0.5 Msun "BH" should be inferred as WD
        star = totest.SingleStar(
            state='BH',
            metallicity=1.0,
            mass=0.5,
        )
        assert star.state == 'WD'

    def test_uncaught_state_warns(self):
        # Covers lines 265-270: unknown state triggers warning
        with warns(match="was not caught"):
            star = totest.SingleStar(
                state='some_unknown_state',
                metallicity=1.0,
                mass=5.0,
            )
        # should still initialize as HMS ZAMS
        expected_X = 1.0 - totest.zams_table[1.0] - 1.0 * totest.Zsun
        assert star.center_h1 == approx(expected_X, abs=1e-10)

    def test_wd_init(self):
        star = totest.SingleStar(
            state='WD',
            metallicity=1.0,
            mass=0.6,
        )
        assert star.state == 'WD'

    def test_hrich_core_he_burning(self):
        # Covers branch 219->223: H-rich Core_He_burning (not stripped_He)
        star = totest.SingleStar(
            state='H-rich_Core_He_burning',
            metallicity=1.0,
            mass=8.0,
        )
        # H-rich Core_He_burning: default_log_R = HIGH_LOGR_GUESS = 4.0
        assert star.log_R == 4.0
        assert star.he_core_mass == 8.0

    def test_kwargs_with_preexisting_scalar_attrs(self):
        # Covers branches 337->339 ... 357->363, 365->364, 371->369, 376->375
        # by passing all optional attrs as kwargs so hasattr is True
        from posydon.grids.SN_MODELS import SN_MODELS
        kwargs = dict(
            state='H-rich_Core_H_burning',
            metallicity=1.0,
            mass=10.0,
            # SN attributes (lines 329-358)
            natal_kick_array=[200.0, 0.5, 1.0, 2.0],
            spin_orbit_tilt_first_SN=0.1,
            spin_orbit_tilt_second_SN=0.2,
            f_fb=0.5,
            SN_type='CCSN',
            m_disk_accreted=0.01,
            m_disk_radiated=0.02,
            h1_mass_ej=1.0,
            he4_mass_ej=2.0,
            M4=1.5,
            mu4=0.3,
            interp1d=None,
            # CE quantities (lines 363-366)
            m_core_CE_1cent=0.5,
            m_core_CE_10cent=0.6,
            m_core_CE_30cent=0.7,
            m_core_CE_pure_He_star_10cent=0.8,
            r_core_CE_1cent=0.1,
            r_core_CE_10cent=0.2,
            r_core_CE_30cent=0.3,
            r_core_CE_pure_He_star_10cent=0.4,
            # He depletion quantities (lines 369-372)
            avg_c_in_c_core_at_He_depletion=0.3,
            co_core_mass_at_He_depletion=1.2,
        )
        # SN model attributes (lines 374-377)
        for model_name in SN_MODELS.keys():
            kwargs[model_name] = 'test_value'

        star = totest.SingleStar(**kwargs)
        assert star.spin_orbit_tilt_first_SN == 0.1
        assert star.f_fb == 0.5
        assert star.m_disk_accreted == 0.01
        assert star.m_core_CE_1cent == 0.5
        assert star.avg_c_in_c_core_at_He_depletion == 0.3


class TestSingleStarRestoreWithHooks:
    """Tests for restore with hooks, covering lines 410-414."""

    def test_restore_with_hooks(self, hms_star):
        hms_star.my_extra_col = [1.0]
        hms_star.mass = 8.0
        hms_star.append_state()
        hms_star.my_extra_col.append(2.0)
        hms_star.mass = 6.0
        hms_star.append_state()
        hms_star.my_extra_col.append(3.0)

        hook = MagicMock()
        hook.extra_star_col_names = ['my_extra_col']

        hms_star.restore(i=1, hooks=[hook])
        assert len(hms_star.my_extra_col) == 2
        assert hms_star.my_extra_col == [1.0, 2.0]
        assert hms_star.mass == 8.0

    def test_restore_with_hook_without_extra_cols(self, hms_star):
        hms_star.mass = 8.0
        hms_star.append_state()

        hook = MagicMock(spec=[])  # no extra_star_col_names attribute
        hms_star.restore(i=0, hooks=[hook])
        assert hms_star.mass == 10.0


class TestSingleStarToDfExtraCoverage:
    """Additional to_df tests for remaining branches."""

    def test_unequal_history_lengths(self, hms_star):
        # Covers lines 484-485: unequal column lengths
        hms_star.mass = 9.0
        hms_star.append_state()
        # Manually make one history shorter to simulate a failed run
        hms_star.log_R_history = [0.0]
        assert len(hms_star.mass_history) == 2
        assert len(hms_star.log_R_history) == 1
        df = hms_star.to_df()
        assert len(df) == 2

    def test_attribute_error_in_to_df(self, hms_star):
        # Covers lines 493-494: missing attribute raises AttributeError
        hms_star.custom_missing_history = [1.0]
        # Request a column that exists but references a nonexistent attribute
        del hms_star.custom_missing_history
        with raises(AttributeError, match="Available attributes"):
            hms_star.to_df(
                extra_columns={'custom_missing_history': 'float64'})

    def test_none_values_replaced(self, hms_star):
        # Covers the None replacement branch
        hms_star.profile_history = [None]
        df = hms_star.to_df(include_profile=True)
        assert np.isnan(df['profile'].iloc[0])

    def test_null_value_custom(self, hms_star):
        hms_star.profile_history = [None]
        df = hms_star.to_df(include_profile=True, null_value=-999.0)
        assert df['profile'].iloc[0] == -999.0


class TestSingleStarToOnelineDfExtraCoverage:
    """Additional to_oneline_df tests for natal_kick_array legacy."""

    def test_natal_kick_array_legacy(self, ns_star):
        # Covers lines 556-579: legacy natal_kick_array handling
        ns_star.natal_kick_velocity = 100.0
        ns_star.natal_kick_azimuthal_angle = 1.5
        ns_star.natal_kick_polar_angle = 0.8
        ns_star.natal_kick_mean_anomaly = 3.0
        # the attribute must exist for hasattr to be True
        ns_star.natal_kick_array = True

        with warns(match="natal_kick_array"):
            df = ns_star.to_oneline_df(
                scalar_names=['natal_kick_array'])
        assert 'natal_kick_array_0' in df.columns
        assert 'natal_kick_array_1' in df.columns
        assert 'natal_kick_array_2' in df.columns
        assert 'natal_kick_array_3' in df.columns
        assert 'natal_kick_velocity' in df.columns
        assert df['natal_kick_velocity'].iloc[0] == 100.0
        assert df['natal_kick_azimuthal_angle'].iloc[0] == 1.5

    def test_scalar_name_not_on_star(self, hms_star):
        # Covers branch 553->552: scalar name not present as attribute
        df = hms_star.to_oneline_df(
            scalar_names=['nonexistent_scalar_attr'])
        assert 'nonexistent_scalar_attr' not in df.columns


class TestSingleStarFromRun:
    """Tests for SingleStar.from_run static method."""

    def test_from_run_no_history(self, mock_single_star_run):
        # Covers from_run with history=False (default)
        star = totest.SingleStar.from_run(mock_single_star_run)
        assert isinstance(star, totest.SingleStar)
        # mass should be the final_value
        assert star.mass == approx(9.5)
        # history should only have 1 entry (no history mode)
        assert len(star.mass_history) == 1

    def test_from_run_with_history(self, mock_single_star_run):
        # Covers from_run with history=True
        star = totest.SingleStar.from_run(mock_single_star_run, history=True)
        assert len(star.mass_history) == 3
        assert star.mass_history[0] == approx(10.0)
        assert star.mass_history[-1] == approx(9.5)

    def test_from_run_metallicity(self, mock_single_star_run):
        star = totest.SingleStar.from_run(mock_single_star_run)
        assert star.metallicity == approx(0.0142)

    def test_from_run_metallicity_missing(self, mock_single_star_run):
        # Covers the AttributeError branch for initial_values (line 623-624)
        del mock_single_star_run.initial_values
        star = totest.SingleStar.from_run(mock_single_star_run)
        assert star.metallicity is None

    def test_from_run_he_depletion_values(self, mock_single_star_run):
        star = totest.SingleStar.from_run(mock_single_star_run)
        assert star.avg_c_in_c_core_at_He_depletion == approx(0.35)
        assert star.co_core_mass_at_He_depletion == approx(1.5)

    def test_from_run_state_computed(self, mock_single_star_run):
        star = totest.SingleStar.from_run(mock_single_star_run)
        assert isinstance(star.state, str)
        assert len(star.state) > 0

    def test_from_run_state_history_with_history(self, mock_single_star_run):
        star = totest.SingleStar.from_run(mock_single_star_run, history=True)
        assert len(star.state_history) == 3
        for s in star.state_history:
            assert isinstance(s, str)

    def test_from_run_with_profile(self, mock_single_star_run):
        star = totest.SingleStar.from_run(
            mock_single_star_run, profile=True)
        assert star.profile is not None
        assert star.profile is mock_single_star_run.final_profile1

    def test_from_run_profile_history(self, mock_single_star_run):
        star = totest.SingleStar.from_run(
            mock_single_star_run, history=True, profile=True)
        assert len(star.profile_history) == 3
        assert star.profile_history[0] is None
        assert star.profile_history[1] is None
        assert star.profile_history[2] is mock_single_star_run.final_profile1

    def test_from_run_none_history(self):
        # Covers line 599-600: history1 is None
        run = MagicMock()
        run.history1 = None
        star = totest.SingleStar.from_run(run)
        assert isinstance(star, totest.SingleStar)

    def test_from_run_unmapped_attrs_are_none(self, mock_single_star_run):
        # 'state' and 'metallicity' are mapped to None in
        # STAR_ATTRIBUTES_FROM_STAR_HISTORY_SINGLE, so their
        # history comes from the None branch (col_history = [None]*n_steps)
        star = totest.SingleStar.from_run(mock_single_star_run, history=True)
        # profile is also None-mapped
        assert all(v is None for v in star.profile_history)

    def test_from_run_he_depletion_non_s1_prefix(self):
        # Covers branch 629->627: He depletion column not prefixed "S1_"
        h_cols = ['star_age', 'star_mass', 'log_R', 'center_h1',
                  'center_he4', 'center_c12', 'surface_h1', 'log_LH',
                  'log_LHe', 'log_Lnuc']
        dt = [(c, '<f8') for c in h_cols]
        history1 = np.zeros(2, dtype=dt)
        history1['star_age'] = [1e6, 5e6]
        history1['star_mass'] = [10.0, 9.5]
        history1['center_h1'] = [0.7, 0.01]
        history1['center_he4'] = [0.27, 0.95]
        history1['surface_h1'] = [0.7, 0.7]
        history1['log_LH'] = [3.5, 2.0]
        history1['log_LHe'] = [-10.0, 3.0]
        history1['log_Lnuc'] = [3.5, 3.0]

        fv_cols = [('S1_' + c, '<f8') for c in h_cols if c != 'star_age']
        # add an at_He_depletion column WITHOUT S1_ prefix
        fv_cols += [('S2_avg_c_in_c_core_at_He_depletion', '<f8')]
        final_values = np.zeros(1, dtype=fv_cols)[0]
        for c in h_cols:
            if c == 'star_age':
                continue
            final_values['S1_' + c] = history1[c][-1]
        final_values['S2_avg_c_in_c_core_at_He_depletion'] = 0.4

        initial_values = np.array([(0.0142,)], dtype=[('Z', '<f8')])[0]

        run = MagicMock()
        run.history1 = history1
        run.final_values = final_values
        run.initial_values = initial_values

        star = totest.SingleStar.from_run(run)
        # S2_ prefixed column should NOT be set on the star
        assert not hasattr(star, 'avg_c_in_c_core_at_He_depletion') \
            or star.avg_c_in_c_core_at_He_depletion is None
