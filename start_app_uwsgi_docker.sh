#!/bin/bash

# File used for deployment in docker container.

# Optional. Switch to environment
# source /usr/local/eeris/eeris/bin/activate

# Make sure mongo is running
pid=`pgrep mongo`
if [ -z "$pid" ]; then
    mongod --fork -f /etc/mongod.conf
    mongo < ini/mini.js
fi

service nginx start

# Launch uwsgi
uwsgi /usr/local/eeris/eeris_nilm/ini/uwsgi.ini --reload-on-exception \
      >> /var/log/uwsgi/eeris_nilm.stdout.log \
      2>> /var/log/uwsgi/eeris_nilm.stderr.log
