@echo off
:: Create and setup environment
call conda create -n NINA_georef_seagul python=3.8 -y
call conda activate NINA_georef_seagul
call pip install .