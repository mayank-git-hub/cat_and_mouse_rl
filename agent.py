import numpy as np

class agent():

	def __init__(self, env, brain):
		
		self.env=env

		self.brain = brain
		
		self.s = 0

		self.clean_start()

	def act(self):
		
		x = np.random.randint(0, len(self.env.poss[self.s]), 1)[0]

		return self.env.poss[self.s][x]

	def train(self, s_):

		s_flat = np.copy(s_)

		s_ = np.unravel_index(s_, [self.env.size, self.env.size], order='C')
		
		self.brain.learn(self.s, s_flat, self.env.maze[s_[0], s_[1]])

	def step(self):

		s_ = self.act()

		self.train(s_)

		if self.env.maze.ravel()[s_] != 0:

			self.clean_start()

		else:

			self.s = s_

	def clean_start(self):

		x = np.where(self.env.maze.ravel() == 0)[0]

		self.s = np.random.choice(x)
