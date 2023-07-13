import sys

import traci
from sumolib import checkBinary

sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "corniche_base.sumocfg", "--device.rerouting.with-taz", "true",
           "--device.rerouting.threads", "6", "--ignore-route-errors", "true", "--collision.check-junctions", "true",
           "--emergencydecel.warning-threshold", "1.1",
           "--scale", "10",             # traffic scaled to 10x
           ]

traci.start(sumoCmd)

# simulate 24 hours (CHANGE START AND END TIME IN sumocfg FIRST)
step = 1
start = 0
stop = 86400

for step in range(start, stop, step):
    traci.simulationStep()

traci.close()
sys.stdout.flush()
