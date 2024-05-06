import unittest
from ExtendedStack import ExtendedStack



class TestExtendedStack(unittest.TestCase):
    def setUp(self):
        self.es = ExtendedStack([1, 2, 3, 4, -3, 3, 5, 10])
    
    def test_div(self):
        self.es.div()
        self.assertEqual(self.es.pop(), 2)
    
    def test_sub(self):
        self.es.sub()
        self.assertEqual(self.es.pop(), 5)
    
    def test_sum(self):
        self.es.sum()
        self.assertEqual(self.es.pop(), 15)
    
    def test_mul(self):
        self.es.mul()
        self.assertEqual(self.es.pop(), 50)


if __name__ == "__main__":
    unittest.main()
