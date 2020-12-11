import models
import calliope
import pdb


def dev_test():
    ts_data = models.load_time_series_data('6_region')
    ts_data = ts_data.loc['2017-01-01']
    model = models.SixRegionModel(ts_data=ts_data,
                                  run_mode='plan')
    pdb.set_trace()


if __name__ == '__main__':
    dev_test()
