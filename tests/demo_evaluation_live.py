"""
Copyright 2020 Christos Diou

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import logging
import sys
from eeris_nilm import evaluation


def print_table(metrics):
    """
    Prepare a table with results
    """
    for name in metrics.keys():
        if not metrics[name]['h_n']:
            continue
        print("%s & %d & %d & %.2f & %.2f & %.2f %.2f\\\\" %
              (name.replace('_', r'\_'), metrics[name]['n'], metrics[name]['d'],
               metrics[name]['pm'], metrics[name]['ps'],
               metrics[name]['pd'], metrics[name]['c']))


if len(sys.argv) > 1:
    house = sys.argv[1]
else:
    house = None

redd_path = 'tests/data/redd'
if house is not None:
    history_file = 'tests/data/redd_live/' + house + '_live_hist.dill'
    metrics = evaluation.live_redd_evaluation(redd_path, house=house,
                                              history_file=history_file)
    evaluation.live_evaluation_print(metrics)
else:
    for h in ['house_1', 'house_2', 'house_3']:
        history_file = 'tests/data/redd_live/' + h + '_live_hist.dill'
        metrics = evaluation.live_redd_evaluation(redd_path, house=h,
                                                  history_file=history_file)
        print(h)
        print_table(metrics)
