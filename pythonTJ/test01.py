import unittest
import sys
sys.path.append('.')
from hw01 import sigmoid


class MyTestCase(unittest.TestCase):
    def test_sigmoid(self):
        z1 = sigmoid(0)
        z2 = sigmoid(100)
        z3 = sigmoid(-100)

        self.assertEqual(z1, 0.5, 'method sigmoid test failed')
        self.assertTrue(abs(1 - z2) < 0.01, 'method sigmoid test failed')
        self.assertTrue(abs(z3) < 0.01, 'method sigmoid test failed')


if __name__ == '__main__':
    unittest.main()
