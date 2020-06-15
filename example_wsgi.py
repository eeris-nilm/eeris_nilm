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

import eeris_nilm.app
import logging
logging.basicConfig(level=logging.DEBUG)

"""
Use thread=False to test with "manual" requests for clustering and
activations (see demo_hart_web_service_1.py).

Otherwise with thread=True a thread is started to perform these operations
periodically (see demo_hart_web_service_2.py).
"""
application = eeris_nilm.app.get_app("ini/eeris.ini", thread=True)
