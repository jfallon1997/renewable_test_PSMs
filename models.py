"""The test case power system models."""


import os
import logging
import shutil
import pandas as pd
import calliope


# Emission intensities of technologies, in ton CO2 equivalent per GWh
EMISSION_INTENSITIES = {'baseload': 200,
                        'peaking': 400,
                        'wind': 0,
                        'solar': 0,
                        'unmet': 0}


def load_time_series_data(model_name):
    """Load demand, wind and solar time series data for model.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'

    Returns:
    --------
    ts_data (pandas DataFrame) : time series data for use in model
    """

    ts_data = pd.read_csv('data/demand_wind_solar.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)

    # If 1_region model, take demand, wind and solar from region 5
    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5',
                                  'solar_region5']]
        ts_data.columns = ['demand', 'wind', 'solar']

    return ts_data


def detect_missing_leap_days(ts_data):
    """Detect if a time series has missing leap days.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : time series
    """

    feb28_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 2)
                                & (ts_data.index.day == 28)]
    feb29_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 2)
                                & (ts_data.index.day == 29)]
    mar01_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 3)
                                & (ts_data.index.day == 1)]
    if len(feb29_index) < min((len(feb28_index), len(mar01_index))):
        return True

    return False


def get_scenario(run_mode, baseload_integer, baseload_ramping, allow_unmet):
    """Get the scenario name for different run settings.

    Parameters:
    -----------
    run_mode (str) : 'plan' or 'operate': whether to let the model
        determine the optimal capacities or work with prescribed ones
    baseload_integer (bool) : activate baseload integer capacity
        constraint (built in units of 3GW)
    baseload_ramping (bool) : enforce baseload ramping constraint
    allow_unmet (bool) : allow unmet demand in planning mode (always
        allowed in operate mode)

    Returns:
    --------
    scenario (str) : name of scenario to pass in Calliope model
    """

    scenario = run_mode
    if run_mode == 'plan' and not baseload_integer:
        scenario = scenario + ',continuous'
    if run_mode == 'plan' and baseload_integer:
        scenario = scenario + ',integer'
    if run_mode == 'plan' and allow_unmet:
        scenario = scenario + ',allow_unmet'
    if baseload_ramping:
        scenario = scenario + ',ramping'

    return scenario


def get_cap_override_dict(model_name, fixed_caps):
    """Create an override dictionary that can be used to set fixed
    fixed capacities in a Calliope model run.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    fixed_caps (pandas Series/DataFrame or dict) : the fixed capacities.
        A DataFrame created via model.get_summary_outputs will work.

    Returns:
    --------
    o_dict (dict) : A dict that can be fed as override_dict into Calliope
        model in operate mode
    """

    if isinstance(fixed_caps, pd.DataFrame):
        fixed_caps = fixed_caps.iloc[:, 0]  # Change to Series

    o_dict = {}

    # Add baseload, peaking, wind and solar capacities
    if model_name == '1_region':
        for tech, attribute in [('baseload', 'energy_cap_equals'),
                                ('peaking', 'energy_cap_equals'),
                                ('wind', 'resource_area_equals'),
                                ('solar', 'resource_area_equals')]:
            idx = f'locations.region1.techs.{tech}.constraints.{attribute}'
            o_dict[idx] = fixed_caps[f'cap_{tech}_total']

    # Add baseload, peaking, wind, solar and transmission capacities
    if model_name == '6_region':
        for region in [f'region{i+1}' for i in range(6)]:
            for tech, attribute in [('baseload', 'energy_cap_equals'),
                                    ('peaking', 'energy_cap_equals'),
                                    ('wind', 'resource_area_equals'),
                                    ('solar', 'resource_area_equals')]:
                try:
                    idx = f'locations.{region}.techs.{tech}_{region}.' \
                          f'constraints.{attribute}'
                    o_dict[idx] = fixed_caps[f'cap_{tech}_{region}']
                except KeyError:
                    pass
            for region_to in [f'region{i+1}' for i in range(6)]:
                tech = 'transmission'
                idx = f'links.{region},{region_to}.techs.{tech}_{region}' \
                      f'_{region_to}.constraints.energy_cap_equals'
                try:
                    o_dict[idx] = \
                        fixed_caps[f'cap_transmission_{region}_{region_to}']
                except KeyError:
                    pass

    if len(o_dict.keys()) == 0:
        raise AttributeError('Override dict is empty. Check if something '
                             'has gone wrong.')

    return o_dict


def calculate_carbon_emissions(generation_levels):
    """Calculate total carbon emissions.

    Parameters:
    -----------
    generation_levels (pandas DataFrame or dict) : generation levels
        for the technologies (baseload, peaking, wind, solar and unmet)
    """

    emissions_tot = (
        EMISSION_INTENSITIES['baseload'] * generation_levels['baseload']
        + EMISSION_INTENSITIES['peaking'] * generation_levels['peaking']
        + EMISSION_INTENSITIES['wind'] * generation_levels['wind']
        + EMISSION_INTENSITIES['solar'] * generation_levels['solar']
        + EMISSION_INTENSITIES['unmet'] * generation_levels['unmet']
    )

    return emissions_tot


def save_html(plot_html: str, fname='output.html', title=''):
    """Save calliope plot to html, including missing javascript fix
    (relevant for Calliope versions such as 0.6.6 where plotting may not
    function properly without this workaround)

    Parameters:
    -----------
    plot_html : calliope html output
    fname : path to output file
    title : (optional) insert html head title
    """
    with open(fname, 'w') as fout:
        fout.write(
            f'<html><head><title>{title}</title>'
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            '</head><body>'
            f'{plot_html}'
            '</body></html>'
        )


def rm_folder(fname: str):
    """Remove a folder, except if folder not found

    Parameters:
    -----------
    fname: path to folder for deletion
    """
    try:
        shutil.rmtree(fname)
    except FileNotFoundError:
        pass


class ModelBase(calliope.Model):
    """Instance of either 1-region or 6-region model."""

    def __init__(self, model_name, ts_data, run_mode,
                 baseload_integer=False, baseload_ramping=False,
                 allow_unmet=False, fixed_caps=None, extra_override=None,
                 run_id=0):
        """
        Create instance of either 1-region or 6-region model.

        Parameters:
        -----------
        model_name (str) : either '1_region' or '6_region'
        ts_data (pandas DataFrame) : time series with time series data.
            It may also contain custom time step weights
        run_mode (str) : 'plan' or 'operate': whether to let the model
            determine the optimal capacities or work with prescribed ones
        baseload_integer (bool) : activate baseload integer capacity
            constraint (built in units of 3GW)
        baseload_ramping (bool) : enforce baseload ramping constraint
        allow_unmet (bool) : allow unmet demand in planning mode (always
            allowed in operate mode)
        fixed_caps (dict or Pandas DataFrame) : fixed capacities as override
        extra_override (str) : name of additional override, to customise
            model. The override should be defined in the relevant model.yaml
        run_id (int) : can be changed if multiple models are run in parallel
        """

        if model_name not in ['1_region', '6_region']:
            raise ValueError('Invalid model name '
                             '(choose 1_region or 6_region)')

        self.model_name = model_name
        self.base_dir = os.path.join('models', model_name)
        self.num_timesteps = ts_data.shape[0]

        # Create scenarios and overrides
        scenario = get_scenario(run_mode, baseload_integer,
                                baseload_ramping, allow_unmet)
        if extra_override is not None:
            scenario = ','.join((scenario, extra_override))
        override_dict = (get_cap_override_dict(model_name, fixed_caps)
                         if fixed_caps is not None else None)

        # Calliope requires a CSV file of the time series data to be present
        # at time of initialisation. This creates a new directory with the
        # model files and data for the model, then deletes it once the model
        # exists in Python
        self._base_dir_iter = self.base_dir + '_' + str(run_id)
        if os.path.exists(self._base_dir_iter):
            shutil.rmtree(self._base_dir_iter)
        shutil.copytree(self.base_dir, self._base_dir_iter)
        ts_data = self._create_init_time_series(ts_data)
        ts_data.to_csv(os.path.join(self._base_dir_iter,
                                    'demand_wind_solar.csv'))
        super().__init__(os.path.join(self._base_dir_iter, 'model.yaml'),
                         scenario=scenario,
                         override_dict=override_dict)
        shutil.rmtree(self._base_dir_iter)

        # Adjust weights if these are included in ts_data
        if 'weight' in ts_data.columns:
            self.inputs.timestep_weights.values = \
                ts_data.loc[:, 'weight'].values

        logging.info('Time series inputs:\n%s', ts_data)
        logging.info('Override dict:\n%s', override_dict)
        if run_mode == 'operate' and fixed_caps is None:
            logging.warning('No fixed capacities passed into model call. '
                            'Will read fixed capacities from model.yaml')

    def _create_init_time_series(self, ts_data):
        """Create time series data for model initialisation."""

        # Avoid changing ts_data outside function
        ts_data_used = ts_data.copy()

        if self.model_name == '1_region':
            expected_columns = {'demand', 'wind', 'solar'}
        elif self.model_name == '6_region':
            expected_columns = {'demand_region2', 'demand_region4',
                                'demand_region5', 'wind_region2',
                                'wind_region5', 'wind_region6',
                                'solar_region2', 'solar_region5',
                                'solar_region6'}
        if not expected_columns.issubset(ts_data.columns):
            raise AttributeError('Input time series: incorrect columns')

        # Detect missing leap days -- reset index if so
        if detect_missing_leap_days(ts_data_used):
            logging.warning('Missing leap days detected in input time series.'
                            'Time series index reset to start in 2020.')
            ts_data_used.index = pd.date_range(start='2020-01-01',
                                               periods=self.num_timesteps,
                                               freq='h')

        # Demand must be negative for Calliope
        ts_data_used.loc[:, ts_data.columns.str.contains('demand')] = (
            -ts_data_used.loc[:, ts_data.columns.str.contains('demand')]
        )

        return ts_data_used

    def preview(self, array_str, head=10, time_idx=True, loc=None, **kwargs):
        """Generate a preview of a Model attribute (eg. inputs.resource)

        Parameters:
        -----------
        array_str (str) : attribute path to array
        head (int) : if non-zero, return head instead of full resource array
        time_idx (bool) : set resource index to timesteps if True
        loc (str) : override loc autosearch if set
        kwargs (dict) : pass arguments to pandas DataFrame

        Returns:
        --------
        resource (pandas DataFrame) : processed collection of input array
        """
        # Access array (split at '.' into nested getattr)
        array = self
        for attrib in array_str.split('.'):
            array = getattr(array, attrib)

        # Identify all possible valid loc keys (labels)
        if not loc:
            loc_keys = [k for k in array.__dir__() if 'loc_' in k]
            help_str = ' Try manually passing loc keyword argument'
            # ensure precisely one loc key found
            assert len(loc_keys) > 0, 'No valid loc keys identified.'+help_str
            assert len(loc_keys) < 2, 'Multiple loc keys identified.'+help_str
            loc = loc_keys[0]

        # set index keyword argument to self.inputs.timesteps
        if time_idx:
            kwargs['index'] = self.inputs.timesteps.values

        # Create resource dict containing array loc 'labels' and array values
        labels = getattr(array, loc).values
        resource = pd.DataFrame(dict(zip(labels, array.values)), **kwargs)

        # Either return header or full DataFrame
        return resource.head(head) if head else resource


class OneRegionModel(ModelBase):
    """Instance of 1-region power system model."""

    def __init__(self, ts_data, run_mode, **arg_parameters):
        # Initialize model from ModelBase parent.
        super().__init__('1_region', ts_data, run_mode, **arg_parameters)

    def get_summary_outputs(self):
        """Create pandas DataFrame of subset of model outputs."""
        # Require model self.run() a priori
        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        # Initialise outputs DataFrame
        outputs = pd.DataFrame(columns=['output'])

        # Annualisation correction factor
        corrfac = (8760/self.num_timesteps)

        # Insert installed capacities
        outputs.loc['cap_baseload_total'] = (
            float(self.results.energy_cap.loc['region1::baseload'])
        )
        outputs.loc['cap_peaking_total'] = (
            float(self.results.energy_cap.loc['region1::peaking'])
        )
        outputs.loc['cap_wind_total'] = (
            float(self.results.resource_area.loc['region1::wind'])
        )
        outputs.loc['cap_solar_total'] = (
            float(self.results.resource_area.loc['region1::solar'])
        )

        # Insert 'unmet' capacity, equal to unmet demand
        outputs.loc['peak_unmet_total'] = \
            float(self.results.carrier_prod.loc['region1::unmet::power'].max())

        # Insert generation levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            outputs.loc[f'gen_{tech}_total'] = corrfac * float(
                (self.results.carrier_prod.loc[f'region1::{tech}::power']
                 * self.inputs.timestep_weights).sum()
            )

        # Insert annualised demand levels
        outputs.loc['demand_total'] = -corrfac * float(
            (self.results.carrier_con.loc['region1::demand_power::power']
             * self.inputs.timestep_weights).sum()
        )

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            {tech: outputs.loc[f'gen_{tech}_total'] for tech in
             ['baseload', 'peaking', 'wind', 'solar', 'unmet']}
        )

        return outputs


class SixRegionModel(ModelBase):
    """Instance of 6-region power system model."""

    def __init__(self, ts_data, run_mode, **arg_parameters):
        # Initialize model from ModelBase parent.
        super().__init__('6_region', ts_data, run_mode, **arg_parameters)

    def get_summary_outputs(self):
        """Create pandas DataFrame of subset of relevant model outputs."""

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        # Initialise outputs DataFrame
        outputs = pd.DataFrame(columns=['output'])

        # Annualisation correction factor
        corrfac = (8760/self.num_timesteps)

        # Insert model outputs at regional level
        for region in [f'region{i+1}' for i in range(6)]:

            # Baseload and peaking capacity
            for tech in ['baseload', 'peaking']:
                cap_loc = f'{region}::{tech}_{region}'
                try:
                    outputs.loc[f'cap_{tech}_{region}'] = float(
                        self.results.energy_cap.loc[cap_loc])
                except KeyError:
                    pass

            # Wind and solar capacity
            for tech in ['wind', 'solar']:
                res_loc = f'{region}::{tech}_{region}'
                try:
                    outputs.loc[f'cap_{tech}_{region}'] = float(
                        self.results.resource_area.loc[res_loc])
                except KeyError:
                    pass

            # Peak unmet demand
            for tech in ['unmet']:
                carrier_loc = f'{region}::{tech}_{region}::power'
                try:
                    outputs.loc[f'peak_unmet_{region}'] = float(
                        self.results.carrier_prod.loc[carrier_loc].max())
                except KeyError:
                    pass

            # Transmission capacity
            for tech in ['transmission']:
                for region_to in [f'region{i+1}' for i in range(6)]:
                    trans_loc = f'cap_transmission_{region}_{region_to}'
                    cap_loc = f'{region}::{tech}_{region}' \
                              f'_{region_to}:{region_to}'
                    # No double counting of links -- one way only
                    if int(region[-1]) < int(region_to[-1]):
                        try:
                            outputs.loc[trans_loc] = float(
                                self.results.energy_cap.loc[cap_loc])
                        except KeyError:
                            pass

            # Baseload, peaking, wind, solar and unmet generation levels
            for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
                carrier_loc = f'{region}::{tech}_{region}::power'
                try:
                    outputs.loc[f'gen_{tech}_{region}'] = corrfac * float(
                        (self.results.carrier_prod.loc[carrier_loc]
                         * self.inputs.timestep_weights).sum())
                except KeyError:
                    pass

            # Demand levels
            carrier_loc = f'{region}::demand_power::power'
            try:
                outputs.loc[f'demand_{region}'] = -corrfac * float(
                    (self.results.carrier_con.loc[carrier_loc]
                     * self.inputs.timestep_weights).sum())
            except KeyError:
                pass

        # Insert total capacities
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'transmission']:
            tech_loc = outputs.index.str.contains(f'cap_{tech}')
            outputs.loc[f'cap_{tech}_total'] = outputs.loc[tech_loc].sum()

        tech_loc = outputs.index.str.contains('peak_unmet')
        outputs.loc['peak_unmet_total'] = outputs.loc[tech_loc].sum()

        # Insert total peak unmet demand -- not necessarily equal to
        # peak_unmet_total. Total unmet capacity sums peak unmet demand
        # across regions, whereas this is the systemwide peak unmet demand
        carriers_prod = self.results.carrier_prod.loc_tech_carriers_prod
        tech_loc = carriers_prod.str.contains('unmet')
        outputs.loc['peak_unmet_systemwide'] = float(
            self.results.carrier_prod.loc[tech_loc].sum(axis=0).max())

        # Insert total annualised generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            tech_loc = outputs.index.str.contains(f'gen_{tech}')
            outputs.loc[f'gen_{tech}_total'] = outputs.loc[tech_loc].sum()

        # Insert total annualised demand levels
        outputs.loc['demand_total'] = \
            outputs.loc[outputs.index.str.contains('demand')].sum()

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            {tech: outputs.loc[f'gen_{tech}_total'] for tech in
             ['baseload', 'peaking', 'wind', 'solar', 'unmet']}
        )

        return outputs


if __name__ == '__main__':
    raise NotImplementedError()
