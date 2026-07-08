@echo off
rem Relaunch the big 6-max blueprint run; --resume continues from the last
rem checkpoint in nlhe6_trainer_state_big.pkl (or starts fresh if absent).
rem 12 buckets from vs-5-opponent EHS rollouts, mirroring the HU big run.
rem eval-every 50k keeps multi-GB state pickles to a sane cadence; the 6-max
rem tree is ~5x the HU infoset count, so expect ~6-10 GB RSS at saturation.
rem The redirection lives *inside* the started cmd: `start prog 1>>log` hands
rem the child a fresh console and the log stays empty.
cd /d "%~dp0"
start "nlhe6-blueprint" /min cmd /c ".venv\Scripts\python.exe nlhe6_blueprint.py --iterations 2000000 --eval-every 50000 --eval-hands 4000 --buckets 12 --save nlhe6_blueprint_big.pkl --state-file nlhe6_trainer_state_big.pkl --resume 1>> nlhe6_train.log 2>> nlhe6_train.err"
echo Training launched (appending to nlhe6_train.log).
