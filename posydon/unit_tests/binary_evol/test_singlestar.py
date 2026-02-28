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

from pytest import approx, fixture, raises, warns

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
