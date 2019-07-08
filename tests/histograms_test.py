import unittest
import cv2
import histograms

class TestHistogram(unittest.TestCase):

    def test_(self):
        testimage = cv2.imread("./images/test64x64-bw.jpg")
        histrogram_H = histograms.hist_H_calculation(testimage)
        self.assertEqual(histrogram_H[0], 64*64,
                         'wrong histogram calculation')


if __name__ == '__main__':
    unittest.main()
