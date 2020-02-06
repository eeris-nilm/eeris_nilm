"""
Copyright 2019 Christos Diou

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

import pandas as pd
from eeris_nilm.algorithms import hart

# TODO:
# 1. Add use-cases including missing values, measurement errors, duplicates, past values,
# errors in sampling rate etc.
# 2. Convert to unit tests

a = '{"active":{"1561543133623":170,"1561543132348":170,"1561543131124":170,"1561543129857":170},"reactive":{"1561543133623":-132.8185565061,"1561543132348":-132.8185565061,"1561543131124":-132.8185565061,"1561543129857":-132.8185565061},"voltage":{"1561543133623":236.5,"1561543132348":236.5,"1561543131124":236.5,"1561543129857":236.3}}'

b = '{"active":{"1561543139841":170,"1561543138558":170,"1561543137333":170,"1561543136101":170},"reactive":{"1561543139841":-132.8185565061,"1561543138558":-132.8185565061,"1561543137333":-132.8185565061,"1561543136101":-132.8185565061},"voltage":{"1561543139841":236.3,"1561543138558":236.3,"1561543137333":236.5,"1561543136101":236.2}}'

c = '{"active":{"1561543646712":130,"1561543645476":130,"1561543644139":130,"1561543642877":130},"reactive":{"1561543646712":-117.0525257587,"1561543645476":-117.0525257587,"1561543644139":-117.0525257587,"1561543642877":-117.0525257587},"voltage":{"1561543646712":237,"1561543645476":237,"1561543644139":237,"1561543642877":237}}'

d = '{"active":{"1339354800000":90.0,"1339354801000":90.0,"1339354802000":90.0,"1339354803000":90.0,"1339354804000":90.0},"reactive":{"1339354800000":-108.5896530151,"1339354801000":-108.5896530151,"1339354802000":-108.5896530151,"1339354803000":-107.5743942261,"1339354804000":-107.5743942261},"voltage":{"1339354800000":237.9000091553,"1339354801000":237.9000091553,"1339354802000":237.9000091553,"1339354803000":237.9000091553,"1339354804000":237.5666656494}}'


a_df = pd.read_json(a)
b_df = pd.read_json(b)
c_df = pd.read_json(c)

model = hart.Hart85eeris(1)
model.update(a_df)
model.update(b_df)
model.update(c_df)
model.update(a_df)
