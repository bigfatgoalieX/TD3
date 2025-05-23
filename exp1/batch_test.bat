@echo off

set LOGFILE=baseline_output.txt

echo ================================ >> %LOGFILE%
echo Run started at %DATE% %TIME% >> %LOGFILE%
echo ================================ >> %LOGFILE%

call python reward_test.py --gravity 1.0 --friction 1.0 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 0.6 --friction 1.0 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 1.0 --friction 2.0 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 1.0 --friction 1.0 --mass 1.5 >>%LOGFILE%
call python reward_test.py --gravity 1.4 --friction 2.0 --mass 1.5 >>%LOGFILE%
call python reward_test.py --gravity 2.0 --friction 1.0 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 1.0 --friction 3.0 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 1.0 --friction 1.0 --mass 2.0 >>%LOGFILE%
call python reward_test.py --gravity 0.5 --friction 0.5 --mass 1.0 >>%LOGFILE%
call python reward_test.py --gravity 1.5 --friction 2.5 --mass 1.6 >>%LOGFILE%

pause
