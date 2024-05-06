class RealString:
	def __init__(self, some_str):
		self.some_str = str(some_str)
 
	def __eq__(self, other):
		if not isinstance(other, RealString):
			other = RealString(other)
		return len(self.some_str) == len(other.some_str)
 
	def __lt__(self, other):
		if not isinstance(other, RealString):
			other = RealString(other)
		return len(self.some_str) < len(other.some_str)
 
	def __le__(self, other):
		return self == other or self < other
