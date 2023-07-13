import os
import sys
import optparse
import subprocess
import random
import time

import numpy as np
#import pdb; pdb.set_trace()
import datetime



import set_sumo_home
import traci
import traci.constants as tc
import sumolib
from sumolib import checkBinary
sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "GHARRAFA_24H.sumocfg","--device.rerouting.with-taz","true","--device.rerouting.threads","6"]
#Port=8813


#traci.connect()

traci.start(sumoCmd)

time.sleep(5)

step = 1
start = 25200
stop = 28800

#traci.init(port=8816,numRetries=10,host='localhost',label='sumotest')

for step in range(start,stop,step):


  traci.simulationStep()
  
  
    

traci.close()
sys.stdout.flush()

