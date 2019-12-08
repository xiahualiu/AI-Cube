# test cube function
from cube import cube
import numpy as np

mycube=cube()
mycube.turn(0)
mycube.turn(3)

cube2=mycube.new_cube(0)

print(mycube.check(mycube.state))

pass