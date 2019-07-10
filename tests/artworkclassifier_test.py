import unittest
import artworkclassifier


class TestVectorized(unittest.TestCase):

    def test_(self):
        awclassifier = artworkclassifier.ArtworkClassifier.fromfile('./images', 'realphoto01.jpg')
        result = awclassifier.isartwork()

        self.assertTrue(result, 'artwork bad classification')


if __name__ == '__main__':
    unittest.main()
