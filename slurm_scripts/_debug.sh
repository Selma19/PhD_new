#!/bin/bash
echo $1
hostname
../venv/bin/python ../src/main_debug2.py $1 $2
