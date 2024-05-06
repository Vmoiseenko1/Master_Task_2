import unittest
from MoneyBox import MoneyBox



class TestMoneyBox(unittest.TestCase):
    def setUp(self):
        self.mb = MoneyBox(100)
        self.mb.add(50)
    
    def test_add(self):
        self.assertEqual(self.mb.can_add(50), True)
        self.assertEqual(self.mb.can_add(51), False)

    def test_retrieve(self):
        self.assertEqual(self.mb.can_retrieve(50), True)
        self.assertEqual(self.mb.can_retrieve(51), False)


if __name__ == "__main__":
    unittest.main()
