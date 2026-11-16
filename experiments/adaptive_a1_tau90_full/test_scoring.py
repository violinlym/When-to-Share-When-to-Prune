import unittest

from score_full import correct, gsm_answer
from scorer_v2 import equivalent, extract_answer


class ScoringTests(unittest.TestCase):
    def test_gsm_final_numeric_answer(self):
        self.assertEqual(gsm_answer('First 1. The answer is 1,234.\nQ: next'), '1234')
        self.assertEqual(gsm_answer('The answer is 3. The answer is -2.'), '-2')
        self.assertEqual(correct('gsm8k','The answer is 1,234.','1234'),('1234',True))
        self.assertEqual(correct('gsm8k','No explicit answer','1234'),(None,False))

    def test_math_existing_equivalence(self):
        self.assertEqual(extract_answer('x=1, finally \\boxed{\\frac{1}{2}}'),r'\frac{1}{2}')
        self.assertTrue(equivalent(r'\frac{1}{2}','0.5'))
        self.assertFalse(correct('math500','No box','0.5')[1])


if __name__=='__main__':unittest.main()
