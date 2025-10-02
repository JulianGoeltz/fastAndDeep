#!python3
import argparse
import json
import numpy as np
from pprint import pprint


json_filename = '/jenkins/results/p_jg_FastAndDeep_oneAlloc/execution_stats.json'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--numberBuilds', default=10, type=int)
    parser.add_argument('--setup', default='all', type=str)
    args = parser.parse_args()

    with open(json_filename, 'r') as f:
        all_data_ = json.load(f)

    all_data = {int(k): v for k, v in all_data_.items()}

    # getting correctly sorted subset of builds
    build_keys = np.array(sorted(
        [i for i in all_data.keys() if (
            (args.setup == 'all' or all_data[i]['HX'] == args.setup)
        )
        ]
    )[-args.numberBuilds:])

    rate_success = np.mean([all_data[key]['success'] == '1' for key in build_keys])
    rate_laststeps = {}
    for step in np.unique([all_data[key]['laststep'] for key in all_data]):
        # only count failed builds
        tmp = np.mean(
            [all_data[key]['laststep'] == step if all_data[key]['success'] == '0' else 0 for key in build_keys])
        if tmp > 0:
            rate_laststeps[step] = tmp

    print(f"using {len(build_keys)} builds")
    print(f"rate of success: {rate_success}")
    print("where did the failures occur:")
    pprint(rate_laststeps)

    rate_setups_train, rate_setups_calib, rate_setups_final = {}, {}, {}
    for setup in np.unique([all_data[key]['HX'] for key in all_data]):
        tmp = np.sum(
            [all_data[key]['HX'] == setup
             for key in build_keys if all_data[key]['laststep'] == 'create calib'])
        if tmp > 0:
            rate_setups_calib[setup] = tmp
        tmp = np.sum(
            [all_data[key]['HX'] == setup
             for key in build_keys if all_data[key]['laststep'] == 'training'])
        if tmp > 0:
            rate_setups_train[setup] = tmp
        tmp = np.sum(
            [all_data[key]['HX'] == setup
             for key in build_keys if (
                 all_data[key]['laststep'] == 'finalisation' and 
                 all_data[key]['success'] == '0')
             ])
        if tmp > 0:
            rate_setups_final[setup] = tmp
    print("distribution of setups that failed during calib:")
    pprint(rate_setups_calib)
    print("distribution of setups that failed during training:")
    pprint(rate_setups_train)
    print("distribution of setups that failed during finalisation:")
    pprint(rate_setups_final)
