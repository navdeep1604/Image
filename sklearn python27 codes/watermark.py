
"""Detect watermark in images.

### Requires

- [Pillow](https://pypi.python.org/pypi/Pillow/2.0.0)
"""
import glob

from classify import MultinomialNB
from PIL import Image



TRAINING_POSITIVE = 'train/watermark/*.jpg'
TRAINING_NEGATIVE = 'train/non watermark/*.jpg'
TEST_POSITIVE = 'assets/watermark/*.jpg'
TEST_NEGATIVE = 'assets/non watermark/*.jpg'

# How many pixels to grab from the top-right of image.
CROP_WIDTH, CROP_HEIGHT = 100, 100
RESIZED = (16, 16)


def get_image_data(infile):
    image = Image.open(infile)
    width, height = image.size
    # left upper right lower
    box = width - CROP_WIDTH, 0, width, CROP_HEIGHT
    region = image.crop(box)
    resized = region.resize(RESIZED)
    data = resized.getdata()
    # Convert RGB to simple averaged value.
    data = [sum(pixel) / 3 for pixel in data]
    # Combine location and value.
    values = []
    for location, value in enumerate(data):
        values.extend([location] * value)
    
       
    return values


def main():
    watermark = MultinomialNB()
    #watermark = xgb.XGBClassifier()
    # Training
    count = 0
    for infile in glob.glob(TRAINING_POSITIVE):
        data = get_image_data(infile)
        watermark.train((data, 'positive'))
        #watermark.fit((data, 'positive'))
        count += 1
        print 'Training', count
    for infile in glob.glob(TRAINING_NEGATIVE):
        data = get_image_data(infile)
        watermark.train((data, 'negative'))
        count += 1
        print 'Training', count
    # Testing
    correct, total = 0, 0
    for infile in glob.glob(TEST_POSITIVE):
        data = get_image_data(infile)
        prediction = watermark.classify(data)
        if prediction.label == 'positive':
            correct += 1
        total += 1
        print 'Testing ({0} / {1})'.format(correct, total)
    for infile in glob.glob(TEST_NEGATIVE):
        data = get_image_data(infile)
        prediction = watermark.classify(data)
        if prediction.label == 'negative':
            correct += 1
        total += 1
        print 'Testing ({0} / {1})'.format(correct, total)
    print 'Got', correct, 'out of', total, 'correct'


if __name__ == '__main__':
    main()