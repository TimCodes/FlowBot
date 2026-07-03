@echo off
rem Relaunch the big HUNL blueprint run; --resume continues from the last
rem checkpoint in hunl_trainer_state_big.pkl (or starts fresh if absent).
cd /d "%~dp0"
start "hunl-blueprint" /min .venv\Scripts\python.exe hunl_blueprint.py ^
  --iterations 2500000 --eval-every 100000 --eval-hands 4000 --buckets 12 ^
  --save hunl_blueprint_big.pkl --state-file hunl_trainer_state_big.pkl --resume ^
  1>> hunl_train.log 2>> hunl_train.err
echo Training launched (appending to hunl_train.log).
