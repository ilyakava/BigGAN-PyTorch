"""Test inception.py

Usage:
CUDA_VISIBLE_DEVICES=0 python test/inception.py
"""

import numpy as np

import inception as iscore

def test_IS_on_images_in_memory():
	imageSize = 32
	n_used_imgs = 100
	
	x = np.random.randint(0,256,size=(n_used_imgs,imageSize,imageSize,3), dtype=np.uint8)
	mis, sis = iscore.get_inception_score(x)
	assert(isinstance(mis, np.float32))
	assert(isinstance(sis, np.float32))
	assert(mis != 0.0)
	assert(sis != 0.0)

if __name__ == '__main__':
	test_IS_on_images_in_memory()
	print('Done test_inception.')
