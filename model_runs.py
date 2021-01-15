"""Power system model runs, using Calliope framework."""


import os
import argparse
import time
import logging
import models
import tests
import pdb


# If we are inside an HPC parallel run, let PRN (parallel run number)
# be the index corresponding to the parallel run. If we are not, we set
# PRN = 0, and code is run in sequence. Note that parallel runs should
# start at 1 and not 0 (NOT via pythonic numbering)
if 'PBS_ARRAY_INDEX' in os.environ:    # i.e. if we are in HPC parallel
    PRN = int(os.environ['PBS_ARRAY_INDEX'])
    RUN_ID = os.environ['PBS_JOBNAME'] + '_' + str(PRN)
else:     # i.e. we are running on laptop or regular bash shell
    PRN = 0
    RUN_ID = 'LAPTOP_' + str(PRN)


def parse_args():
    """Read in model run arguments from bash command."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--ts_data_subset_llim', required=True, type=str)
    parser.add_argument('--ts_data_subset_rlim', required=True, type=str)
    parser.add_argument('--run_mode', required=True, type=str)
    parser.add_argument('--logging_level', required=False, type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'], default='WARNING',
                        help='Python logging module verbosity level')
    args = parser.parse_args()

    return args


def get_output_directory(create_directory=False):
    """Get the directory path for run outputs."""
    args = parse_args()

    base_directory = 'outputs_hpc'
    model_directory = args.model_name
    ts_directory = args.ts_data_subset_llim + '_' + args.ts_data_subset_rlim
    output_directory = os.path.join(base_directory,
                                    model_directory,
                                    ts_directory)

    # Create the right output directory if it does not exist
    if create_directory:
        sub_directory_path = ''
        for sub_directory in [base_directory, model_directory, ts_directory]:
            sub_directory_path = os.path.join(sub_directory_path,
                                              sub_directory)
            if not os.path.exists(sub_directory_path):
                os.makedirs(sub_directory_path, exist_ok=True)

    return output_directory


def get_runid_string(baseload_integer, baseload_ramping, allow_unmet):
    """Get the string identifying a model run"""
    args = parse_args()
    runid_string = '_'.join((args.run_mode,
                             'baseload-integer-' + str(baseload_integer),
                             'baseload-ramping-' + str(baseload_ramping),
                             'allow-unmet-' + str(allow_unmet)))
    return runid_string


def import_time_series_data(model_name, ts_subset_llim, ts_subset_rlim):
    """Import time series data for model, without any time slicing."""
    if model_name == '1_region':
        ts_data = models.load_time_series_data(model_name='1_region')
    elif model_name == '6_region':
        ts_data = models.load_time_series_data(model_name='6_region')
    else:
        raise NotImplementedError()
    ts_data = ts_data.loc[ts_subset_llim:ts_subset_rlim]

    return ts_data


def run_simulation(model_name, ts_data, run_mode, baseload_integer,
                   baseload_ramping, allow_unmet, run_id, save_csv=False):
    """Run Calliope model with demand & wind data.

    Returns:
    --------
    results: pandas DataFrame with model outputs
    """

    # extra_override = 'gurobi' if 'LAPTOP' in run_id else None
    extra_override = None

    start = time.time()
    if model_name == '1_region':
        model = models.OneRegionModel(ts_data=ts_data,
                                      run_mode=run_mode,
                                      baseload_integer=baseload_integer,
                                      baseload_ramping=baseload_ramping,
                                      allow_unmet=allow_unmet,
                                      extra_override=extra_override,
                                      run_id=run_id)
    elif model_name == '6_region':
        model = models.SixRegionModel(ts_data=ts_data,
                                      run_mode=run_mode,
                                      baseload_integer=baseload_integer,
                                      baseload_ramping=baseload_ramping,
                                      allow_unmet=allow_unmet,
                                      extra_override=extra_override,
                                      run_id=run_id)
    else:
        raise ValueError('Invalid model name.')

    # Run model and save results
    model.run()
    if save_csv:
        model.to_csv(save_csv)
    finish = time.time()
    if model_name == '1_region':
        tests.test_output_consistency_1_region(model)
    elif model_name == '6_region':
        tests.test_output_consistency_6_region(model)
    results = model.get_summary_outputs()
    results.loc['time'] = finish - start

    return results


def conduct_model_run(baseload_integer, baseload_ramping, allow_unmet):
    """Conduct a single model run.

    Parameters:
    -----------
    iteration (int): starts at 0 for samples and 1980 for years.
    """

    # Read in command line arguments
    args = parse_args()

    # Log the run characteristics
    logging.info('%s', args)

    ts_data = import_time_series_data(args.model_name,
                                      args.ts_data_subset_llim,
                                      args.ts_data_subset_rlim)

    summary_outputs = run_simulation(model_name=args.model_name,
                                     ts_data=ts_data,
                                     run_mode=args.run_mode,
                                     baseload_integer=baseload_integer,
                                     baseload_ramping=baseload_ramping,
                                     allow_unmet=allow_unmet,
                                     run_id=RUN_ID,
                                     save_csv=False)

    return summary_outputs


def conduct_model_runs():
    """Conduct the model runs in HPC."""

    # Read in command line arguments
    args = parse_args()
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=getattr(logging, args.logging_level),
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Create output directory if it does not exist yet
    output_directory = get_output_directory(create_directory=True)

    for baseload_integer in [False, True]:
        for baseload_ramping in [False, True]:
            for allow_unmet in [False, True]:
                output_path = os.path.join(
                    output_directory,
                    get_runid_string(baseload_integer,
                                     baseload_ramping,
                                     allow_unmet) + '.csv'
                    )
                if not os.path.isfile(output_path):
                    summary_outputs = conduct_model_run(baseload_integer,
                                                        baseload_ramping,
                                                        allow_unmet)
                    summary_outputs.to_csv(output_path)
                else:
                    logging.info('Output file already exists.')
                logging.info('\n\n\n\n')


if __name__ == '__main__':
    conduct_model_runs()
