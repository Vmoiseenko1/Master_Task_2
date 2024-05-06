import unittest
from RealString import RealString



class TestMoneyBox(unittest.TestCase):
    def setUp(self):
        self.str1 = RealString('Молоко')
        self.str2 = RealString('Абрикосы растут')
        self.str3 = 'Золото'
        self.str4 = [1, 2, 3]
    
    def test_1(self):
        self.assertEqual(self.str1 < self.str4, True)

    def test_2(self):
        self.assertEqual(self.str1 >= self.str2, False)
    
    def test_3(self):
        self.assertEqual(self.str1 == self.str3, True)


if __name__ == "__main__":
    unittest.main()
