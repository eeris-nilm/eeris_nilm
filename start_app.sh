#!/bin/bash

# Flask
# export FLASK_APP=app.py
# export FLASK_ENV=development
# flask run

# Falcon
gunicorn --reload 'eeris_nilm.app:get_app()'
