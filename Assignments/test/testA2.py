import sys
sys.path.append("..")

import nbimporter
import Assignment2 as a1
import unittest
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

class TestNotebook(unittest.TestCase):

    def setUp(self):
        self.array1 = np.linspace(1, 20, 60)
        self.array2 = np.random.randn(60)
        self.array3 = np.random.randn(60)
        self.mse1 = np.mean((self.array1-self.array2)**2)
        self.mse2 = np.mean((self.array2-self.array3)**2)
        self.mse3 = np.mean((self.array1-self.array3)**2)
        self.img = mpimg.imread("images/cvml.png")[:,:,:3]
        self.sigmoid_ = 1/(1+np.exp(-self.array2))
        self.green = self.array2.reshape(6, 10)
        self.r = np.linspace(1, 10, 810000).reshape((500,1620))
        self.g = np.linspace(1, 10, 810000).reshape((900,900))

    def test_mse(self):
        self.assertEqual(a1.mean_square_error(self.array1, self.array2), self.mse1)
        self.assertEqual(a1.mean_square_error(self.array2, self.array3), self.mse2)
        self.assertEqual(a1.mean_square_error(self.array1, self.array3), self.mse3)

    def test_no_2(self):
        self.assertEqual(a1.stack_channels(self.r, self.g, self.g).shape, (900,900,3))

    def test_no_3(self):
        # Test Sigmoid
        self.assertTrue(np.all(a1.sigmoid(self.array2)==self.sigmoid_))

    def test_no_4(self):
        img_ = a1.read_img("images/cvml.png")
        green_ = a1.get_green_channel(img_)
        resized = a1.resize_img(green_, 500, 600)
        resized_ = a1.resize_img(green_, 350, 600)

        rotated = a1.rotate_img(resized, 90, scale=0.5)
        rotated_ = a1.rotate_img(resized_, 45, 1.2)

        self.assertEqual(rotated.shape, resized.shape)
        self.assertEqual(rotated_.shape, resized_.shape)

        # Test read_img function
        self.assertEqual(img_.shape, (900, 900, 4))
        self.assertTrue(np.all(img_[:,:,:3]==self.img))

        # Test get_green function
        self.assertTrue(np.all(green_==self.img[:,:,1]))
        self.assertEqual(green_.shape, (900,900))

        # Test resize function
        self.assertEqual(resized.shape, (500, 600))
        self.assertEqual(resized_.shape, (350, 600))
        # Test rotate function
        self.assertEqual(np.sum(rotated[:100,:])+np.sum(rotated[400:,:]), 250.0)
        self.assertEqual(rotated_[100, 100], 1.0)

    def test_no_5(self):
        # Test dim output
        self.assertEqual(a1.stack_images(self.green, self.green).shape, (6, 20))

if __name__ == '__main__':
    main = TestNotebook()

    # This executes the unit test/(itself)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNotebook)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)
