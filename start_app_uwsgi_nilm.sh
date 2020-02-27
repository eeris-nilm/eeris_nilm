#!/bin/bash
# uwsgi -c uwsgi.ini --reload-on-exception
uwsgi ini/uwsgi_nilm.ini --reload-on-exception
