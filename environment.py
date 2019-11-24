import numpy as np

class env():
	
	def __init__(self,size=4,cat_r=[-10], cheese_r=[10]):

		self.maze = np.zeros((size,size))
		self.poss2d = []
		
		self.size=size
		
		self.poss=[]

		self.num_cats = len(cat_r)

		self.num_cheese = len(cheese_r)

		self.cat_r = cat_r

		self.cheese_r = cheese_r

		self.cheese_add()

		self.cat_add()

		self.find_poss()

	def find_poss(self):

		for x in range(0,self.size):
			for y in range(0,self.size):
				self.poss2d.append([(x,y-1),(x+1,y),(x,y+1),(x-1,y)])

		for i in range(0,len(self.poss2d)):
			self.poss.append([])
			for e in self.poss2d[i]:
				if e[0] in range(0,self.size) and e[1] in range(0,self.size):
					self.poss[i].append(self.size*e[0]+e[1])

	def cheese_add(self):

		x = np.where(self.maze.ravel() == 0)[0]
		rand = np.random.choice(x, self.num_cheese, replace=False)
		self.maze.ravel()[rand] = self.cheese_r

	def cat_add(self):

		x = np.where(self.maze.ravel() == 0)[0]
		rand = np.random.choice(x, self.num_cats, replace=False)
		self.maze.ravel()[rand] = self.cat_r
