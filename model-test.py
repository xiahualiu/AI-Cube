from model import RL
from model import ExploreMemory

net=RL([7,24],12)
memory=ExploreMemory()
memory.play(3)
memory.process(net)
memory.play(1)
memory.process(net)