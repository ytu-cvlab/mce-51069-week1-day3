import sys
sys.path.append("../")

import nbimporter
import Assignment1 as a1
import unittest
import numpy as np
import matplotlib.image as mpimg
import cv2

class TestNotebook(unittest.TestCase):
    
    currentResult = None # holds last result object passed to run method
    
    def setUp(self):
        self.h = 200
        self.w = 250
        self.img_array = mpimg.imread('images/cvml.png')
        self.green = self.img_array[:,:,1]
        self.resized = cv2.resize(self.green, (self.w, self.h))

    @classmethod
    def setResult(cls, amount, errors, failures, skipped):
        cls.amount, cls.errors, cls.failures, cls.skipped = \
            amount, errors, failures, skipped

    def tearDown(self):
        amount = self.currentResult.testsRun
        errors = self.currentResult.errors
        failures = self.currentResult.failures
        skipped = self.currentResult.skipped
        self.setResult(amount, errors, failures, skipped)

    @classmethod
    def tearDownClass(cls):
        print("\ntests run: " + str(cls.amount))
        print("errors: " + str(len(cls.errors)))
        print("failures: " + str(len(cls.failures)))
        print("success: " + str(cls.amount - len(cls.errors) - len(cls.failures)))
        print("skipped: " + str(len(cls.skipped)))
        
    def run(self, result=None):
        self.currentResult = result # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method        

    # Question 1
    def test_read_img(self):
        img_array_ = a1.read_img('images/cvml.png')
        self.assertTrue(isinstance(img_array_, np.ndarray))
        self.assertEqual(len(img_array_.shape), 3)

    def test_get_green_channel(self):
        green_ = self.img_array[:, :, 1]
        self.assertEqual(len(green_.shape), 2)
        self.assertTrue(isinstance(green_, np.ndarray))


    def test_resize_img(self):
        resized_ = a1.resize_img(self.green, self.h, self.w)
        self.assertEqual(resized_.shape, (self.h, self.w))
        self.assertTrue(isinstance(resized_, np.ndarray))
        self.assertEqual(len(resized_.shape), 2)


    def test_rotated_img(self):
        rotated = a1.rotate_img(self.resized, 90)
        self.assertEqual(rotated.shape, (self.h, self.w))
        self.assertTrue(isinstance(rotated, np.ndarray))
        self.assertEqual(len(rotated.shape), 2)


if __name__ == '__main__':
    main = TestNotebook()

    # This executes the unit test/(itself)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNotebook)
    unittest.TextTestRunner(verbosity=2,stream=sys.stderr).run(suite)
