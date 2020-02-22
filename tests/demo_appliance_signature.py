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
import os.path
import pickle
from eeris_nilm.datasets import redd
from eeris_nilm.datasets import eco
from eeris_nilm.appliance import Appliance
# TODO: Convert to unit tests

# Test with active power only using redd
p = 'tests/data/house_1'
f = 'tests/data/redd_test.pickle'
if os.path.isfile(f):
    with open(f, 'rb') as fp:
        data, labels = pickle.load(fp)
else:
    data, labels = redd.read_redd(p)

app = []
for i in range(3, len(labels)):
    a = Appliance(i, labels.loc[i] + str(i), labels.loc[i])
    a.signature_from_data(data[i])
    app.append(a)
