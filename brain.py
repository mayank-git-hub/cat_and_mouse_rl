import numpy as np

class brain():

	def __init__ (self, size, gamma, l_r):

		self.q_mat = np.zeros([size*size, size*size])
		self.gamma = gamma
		self.l_r = l_r
		self.biggest_number = 1e8
		self.size = size

	def learn(self, s, s_, r):

		# print(s, s_, 'here')

		self.q_mat[s, s_] = self.l_r*(r + self.gamma*np.max(self.q_mat[s_, :])) + (1 - self.l_r)*(self.q_mat[s, s_])

	def exploit(self, s, poss):

		imposs = list(np.arange(self.size*self.size))
		
		for i in poss[s]:
			
			imposs.remove(i)

		imposs = np.array(imposs)

		imposs_num = np.zeros([self.size*self.size])

		imposs_num[imposs] = 1

		s_ = np.argmax(self.q_mat[s, :] - self.biggest_number*imposs_num)

		return s_









