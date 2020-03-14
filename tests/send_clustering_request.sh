#!/bin/bash

# Run this to force clustering on an installation
INST_ID=1
http POST http://localhost:8000/nilm/${INST_ID}/clustering
