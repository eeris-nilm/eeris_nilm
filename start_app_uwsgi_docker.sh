#!/bin/bash

# Optional. Switch to environment
# source /usr/local/eeris/eeris/bin/activate

# Make sure mongo is running
pid=`pgrep mongo`
if [ -z "$pid" ]; then
    mongod --fork -f /etc/mongod.conf
    mongo < ini/mini.js
fi

# Launch uwsgi
uwsgi /usr/local/eeris/eeris_nilm/ini/uwsgi.ini --reload-on-exception
