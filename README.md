




---- REAPER ----

The reaper .py is the kill switch for the main loop. It uses a watchdog to always run in the backround. Reaper checks for errors, heartbeat(loop consistancy), keeps it in a life span of a set number of loops, checks to see if the memory slope is rising too fast, makes sure the cpu isnt starved, etc. 

The reaper when one of the cercumstances above become true, shut down the program. 

---- MEMORY ----

