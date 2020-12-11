import models
import tests
import warnings
import pdb


def dev_test():
    """I develop code here!"""

    model_name = '1_region'
    ts_subset_llim = '2017-01'
    ts_subset_rlim = '2017-01'
    run_mode = 'plan'
    baseload_integer = True
    baseload_ramping = True

    ts_data = models.load_time_series_data(model_name=model_name)
    ts_data = ts_data.loc[ts_subset_llim:ts_subset_rlim]

    if model_name == '1_region':
        model = models.OneRegionModel
        test_output_consistency = tests.test_output_consistency_1_region
    elif model_name == '6_region':
        model = models.SixRegionModel
        test_output_consistency = tests.test_output_consistency_6_region

    model = model(ts_data=ts_data,
                  run_mode=run_mode,
                  baseload_integer=baseload_integer,
                  baseload_ramping=baseload_ramping)
    model.run()
    test_output_consistency(model)
    print(model.get_summary_outputs())


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    dev_test()
