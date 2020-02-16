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

from eeris_nilm import evaluation
import logging
# from eeris_nilm import evaluation

logging.basicConfig(level=logging.DEBUG)

# Load data
redd_path = 'tests/data/'
evaluation.hart_redd_evaluation(redd_path, house='house_1_short',
                                date_start='2011-04-18T00:00',
                                date_end='2011-04-21T23:59')
