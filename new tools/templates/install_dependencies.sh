#!/bin/bash

# Check and install system dependencies
python3 -m finetune.dependency_checker

# Start web server
cd web_ui && flask run