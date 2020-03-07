#!/bin/bash

# Falcon
gunicorn --reload 'eeris_nilm.app:get_app()'
