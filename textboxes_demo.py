import numpy as np
import matplotlib
from PIL import Image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import xml.dom.minidom
import riseml.server
from io import BytesIO
# %matplotlib inline
import caffe
import sys

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = 'examples/TextBoxes/deploy.prototxt'
model_weights = 'TextBoxes_icdar13.caffemodel'

scales=((700,700),)
# IMPORTANT: If use mutliple scales in the paper, you need an extra non-maximum superession for the results
# scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)

def identify_boxes(input_image):
	input_image = Image.open(BytesIO(input_image))
	image = np.asarray(input_image.convert('RGB'), dtype=np.float32)
	image = image/255
	image_height,image_width,channels=image.shape
	for scale in scales:
		start = time.time()
		image_resize_height = scale[0]
		image_resize_width = scale[1]
		transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_mean('data', np.array([104,117,123])) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
		
		net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)		
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image

		# Forward pass.
		detections = net.forward()['detection_out']
		print("Network took %.1f ms" % ((time.time() - start) * 1000))
		start_draw = time.time()
		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

		top_conf = det_conf[top_indices]
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		plt.clf()
		plt.imshow(image)
		currentAxis = plt.gca()
		
		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			xmin = max(1,xmin)
			ymin = max(1,ymin)
			xmax = min(image.shape[1]-1, xmax)
			ymax = min(image.shape[0]-1, ymax)
			score = top_conf[i]
			result=str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)+'\r\n'
			
			name = '%.2f'%(score)
			coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
			color = 'b'
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})

	buf = BytesIO()
	plt.savefig(buf, format='jpg')
	buf.seek(0)
	print("Drawing boxes took %.1f ms" % ((time.time() - start_draw) * 1000))
	print("Total processing time: %.1f ms" % ((time.time() - start) * 1000))
	return buf.getvalue()
riseml.server.serve(identify_boxes, port=os.environ.get('PORT'))
