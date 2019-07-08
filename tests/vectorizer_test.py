import unittest
import vectorizer
import vectorfile


class TestVectorized(unittest.TestCase):

    def test_(self):
        vectorgenerator = vectorizer.Vectorizer('./images', 'artwork01.jpg')
        image_q, general_contours, first_pixel = vectorgenerator.process()
        vectorfile.savetosvg(image_q,general_contours, first_pixel, 'testOutput')
        self.assertEqual(len(general_contours),  63,
                         'wrong number of contours')


if __name__ == '__main__':
    unittest.main()
