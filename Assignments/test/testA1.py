import sys
sys.path.append("../")

import nbimporter
import Assignment1 as a1
import unittest
import numpy as np
import matplotlib.image as mpimg
import cv2


class TestNotebook(unittest.TestCase):
    
    def setUp(self):
        self.h = 200
        self.w = 250
        self.img = mpimg.imread("images/cvml.png")
        self.green = self.img[:,:,1]
        self.resized = cv2.resize(self.green, (self.w, self.h))
        M = cv2.getRotationMatrix2D((self.w//2, self.h//2), 90, 1)
        self.rotated = cv2.warpAffine(self.resized, M, (self.w, self.h))


    # Question 1
    def test_read_img(self):
        # check function
        img_array = a1.read_img("images/cvml.png")
        # check type
        self.assertTrue(isinstance(img_array, np.ndarray))
        # check shape
        self.assertEqual(len(img_array.shape), 3)

    def test_get_green_channel(self):
        # check function
        green = a1.get_green_channel(self.img)
        # check type
        self.assertTrue(isinstance(green, np.ndarray))
        # check shape
        self.assertEqual(len(green.shape), 2)
        # check image
        self.assertTrue(np.all(green==self.green))

    def test_resize_img(self):
        # check function
        resized = a1.resize_img(self.green, self.h, self.w)
        # check shape
        self.assertEqual(resized.shape, (self.h, self.w))
        # check dtype
        self.assertTrue(isinstance(resized, np.ndarray))


    def test_rotated_img(self):
        rotated = a1.rotate_img(self.resized, 90)
        self.assertEqual(rotated.shape, (self.h, self.w))
        self.assertTrue(isinstance(rotated, np.ndarray))
        self.assertEqual(len(rotated.shape), 2)
        self.assertTrue(np.all(rotated==self.rotated))


if __name__ == '__main__':
    main = TestNotebook()

    # This executes the unit test/(itself)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNotebook)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)
