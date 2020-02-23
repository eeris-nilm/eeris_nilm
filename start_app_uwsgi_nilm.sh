#!/bin/bash
# uwsgi -c uwsgi.ini --reload-on-exception
uwsgi conf/uwsgi_nilm.ini --reload-on-exception
