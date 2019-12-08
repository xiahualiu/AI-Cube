# test cube function
from cube import cube
import numpy as np

mycube=cube()
mycube.turn(0)
mycube.turn(3)
print(mycube.check(mycube.state))

pass