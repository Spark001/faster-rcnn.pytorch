import os
import sys
import scipy.io as sio
sys.path.insert(0, '/home/disk/workspace')
from Library.pascal_voc_io import PascalVocWriter
from Library.draw_lib import DrawLib
import cv2
import pdb
import h5py
from tqdm import tqdm
from datasets import ds_utils
import xml.etree.ElementTree as ET

# options = {
# 	'SS':selectivesearch,
# 	'EB':edgebox
# }

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # if obj_struct['name'] not in ['TLS', 'OL', 'TLF']:
        #     continue
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        #  obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        tmp = bbox.find('score')
        if tmp:
            obj_struct['score'] = float(tmp.text)
        else:
            obj_struct['score'] = 0


        objects.append(obj_struct)

    return objects


def selectivesearch():

	pass


def combine():
	pass


class GenerateGT:
	def __init__(self):
		Unsupervised = None
		CAM = None


def mat2xml(matname, img_dir, anno_dir):
	if not os.path.exists(anno_dir):
		os.mkdir(anno_dir)
	Data = sio.loadmat(matname)
	imglist = Data['images'].ravel()
	# print (imglist)
	raw_data = Data['boxes'].ravel()
	box_list = []
	for i in range(raw_data.shape[0]):
		boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
		# keep = ds_utils.unique_boxes(boxes)
		# boxes = boxes[keep, :]
		# keep = ds_utils.filter_small_boxes(boxes, 20)
		# boxes = boxes[keep, :]
		box_list.append(boxes)
	for ind, boxes in tqdm(enumerate(box_list)):
		imgname = str(imglist[ind].ravel()[0])
		imgpath = os.path.join(img_dir, imgname + '.jpg')
		if not os.path.exists(imgpath):
			print(imgpath,'not exist!')
			continue
		annfile = os.path.join(anno_dir, imgname + '.xml')
		img = cv2.imread(imgpath)
		height, width, _ = img.shape
		writer = PascalVocWriter(img_dir, imgname, img.shape)
		norm_box = False
		# print (len(boxes))
		for box in boxes:
			x1, y1, x2, y2, score = float(box[0]),\
									float(box[1]),\
									float(box[2]),\
									float(box[3]),\
									-1
			if norm_box:
				x1 *= width
				x2 *= width
				y1 *= height
				y2 *= height
			x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
			cls = 'objectness'
			diff = 0
			writer.addBndBox(x1, y1, x2, y2, cls, diff, score=score)
		writer.save(annfile)
		# break


def edgebox2xml(filename, img_dir, anno_dir):
	if not os.path.exists(anno_dir):
		os.mkdir(anno_dir)
	edgebox = h5py.File(filename, 'r')
	imgnames = edgebox.keys()
	print (len(imgnames))
	tol = 0
	norm_box = True
	for imgname in tqdm(imgnames):
		imgpath = os.path.join(img_dir, imgname+'.jpg')

		annfile = os.path.join(anno_dir, imgname+'.xml')
		if not os.path.exists(imgpath):
			continue
		img = cv2.imread(imgpath)
		height,width,_ = img.shape
		print (imgpath, img.shape)
		writer = PascalVocWriter(img_dir, imgname, img.shape)
		for _,i in enumerate(edgebox[imgname]):
			if _ > 20:break
			# print(i)
			# if float(i[-1]) < 0.1: continue
			x1,y1,x2,y2,score = float(i[0]),float(i[1]),float(i[2]),float(i[3]),float(i[4])
			if norm_box:
				x1 *= width
				x2 *= width
				y1 *= height
				y2 *= height
			x1,y1,x2,y2 = map(round, [x1,y1,x2,y2])
			cls = 'objectness'
			diff = 0
			writer.addBndBox(x1,y1,x2,y2,cls, diff, score=score)
		writer.save(annfile)
		tol += 1
		if tol >10:
			break


def iou(box1,box2):
	xmin = max(box1[0], box2[0])
	ymin = max(box1[1], box2[1])
	xmax = min(box1[2], box2[2])
	ymax = min(box1[3], box2[3])
	w = max(xmax - xmin,0)
	h = max(ymax - ymin,0)
	inters = w*h
	union = (box1[2]-box1[0]+1)*(box1[3]-box1[1]) + \
			(box2[2]-box2[0]+1)*(box2[3]-box2[1]) - \
			inters
	overlap = inters/union
	return overlap


def match(gt, method,filename = '',threshold = 0.5):
	'''
	Check the matching degree groundtruth and the objectness generated from method
	:param gt: groundtruth [[x1,y1,x2,y2,cls]]
	:param method: the objectness generated from method [[x1,y1,x2,y2,cls,score]]
	:param corresponding filename:
	:return: Dict
	'''

	method = sorted(method,key=lambda bbox:-bbox[-1])# sort error
	recall = [0]*(len(method)+1)
	j = 0
	recall[0] = 0
	while j < len(method):
		methodbox = method[j]
		recall[j+1] = recall[j]
		for ind, gtbox in enumerate(gt):
			if iou(gtbox[:-1], methodbox[:-2]) > threshold:
				recall[j+1] += 1
				gt.remove(gtbox)
				break
		j += 1
	return recall


def cal_recall():
	def load_gt(annofile):
		tree = ET.parse(annofile)
		objs = tree.findall('object')
		boxes = []
		for obj in objs:
			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = float(bbox.find('xmin').text) - 1
			y1 = float(bbox.find('ymin').text) - 1
			x2 = float(bbox.find('xmax').text) - 1
			y2 = float(bbox.find('ymax').text) - 1
			cls = obj.find('name').text.lower().strip()
			boxes.append([x1,y1,x2,y2,cls])
		return boxes
	def load_ss(annofile):
		tree = ET.parse(annofile)
		objs = tree.findall('object')
		boxes = []
		for obj in objs:
			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = float(bbox.find('xmin').text)
			y1 = float(bbox.find('ymin').text)
			x2 = float(bbox.find('xmax').text)
			y2 = float(bbox.find('ymax').text)
			cls = obj.find('name').text.lower().strip()
			boxes.append([x1, y1, x2, y2, cls, 0])
		# print(boxes)
		return boxes
	def load_eb(annofile):
		tree = ET.parse(annofile)
		objs = tree.findall('object')
		boxes = []
		for obj in objs:
			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = float(bbox.find('xmin').text) - 1
			y1 = float(bbox.find('ymin').text) - 1
			x2 = float(bbox.find('xmax').text) - 1
			y2 = float(bbox.find('ymax').text) - 1
			cls = obj.find('name').text.lower().strip()
			score = float(bbox.find('score').text)
			boxes.append([x1, y1, x2, y2, cls, score])
		return boxes
	def lower_bound(nums, target):
		low, high = 0, len(nums) - 1
		pos = len(nums)
		while low < high:
			mid = (low + high) // 2
			if nums[mid] < target:
				low = mid + 1
			else:  # >=
				high = mid
				pos = high
		return pos

	writed = True
	if writed:
		import pickle
		import matplotlib.pyplot as plt
		with open('./vis_ss_train.pkl', 'rb') as f:
			recall = pickle.load(f)
			pos = lower_bound(recall, recall[-1])
			plt.scatter(pos, recall[-1], marker='x',s=50)
			plt.text(pos,recall[-1], (pos, round(recall[-1],4)),ha='center', va='bottom', fontsize=10)
			plt.plot(range(len(recall)), recall, color="blue", linewidth=2.5, linestyle="-", label="SelectiveSearch")
			plt.xlabel("proposal number")
			plt.ylabel("recall")
			# DrawLib.drawPlot(range(len(recall)), "proposal number",
			# 				recall, "recall",
			# 				)

		with open('./vis_eb_train_sorted.pkl', 'rb') as f:
			recall = pickle.load(f)
			pos = lower_bound(recall, recall[-1])
			plt.text(pos,recall[-1], (pos, round(recall[-1],4)),ha='center', va='bottom', fontsize=10)
			plt.scatter(pos, recall[-1], marker='x', s=50)
			plt.plot(range(len(recall)), recall, color="red", linewidth=2.5, linestyle="-", label='EdgeBox')
			# DrawLib.drawPlot(range(len(recall)), "proposal number",
			# 				recall, "recall",
			# 				)
		plt.legend(loc='lower right')
		# plt.legend(loc='upper left')
		plt.title("Recall / Proposal Method")
		plt.show()
		return
	gt_annodir = '/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/Annotations'
	ss_annodir = '/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/SSAnnotation'
	eb_annodir = '/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/EBAnnotation'
	imglist = os.listdir(eb_annodir)
	length = 20000
	recall = [0]*length
	gt_num = 0
	tol = 0
	for file in tqdm(imglist):
		# if tol>=00:
		# 	break
		tol += 1
		# if tol % 10 == 0:
		# 	print ("{}/{}".format(tol,len(imglist)))
		gt = load_gt(os.path.join(gt_annodir, file))
		gt_num += len(gt)
		# print(len(gt))
		# ss = load_ss(os.path.join(ss_annodir, file))
		# rc = match(gt, ss, filename=file)
		eb = load_eb(os.path.join(eb_annodir, file))
		rc = match(gt, eb, filename=file)
		# print(rc[-1])
		for num in range(1,length):
			if num < len(rc):
				# print (num)
				recall[num] += rc[num]
			else:
				recall[num] += rc[-1]
	gt_num *= 1.0
	def div(x):
		x /= gt_num
	# recall = map(div, recall)
	recall = [i/gt_num for i in recall]

	pos = lower_bound(recall, recall[-1])
	print(recall[-1],'/',pos)
	if not writed:
		import pickle
		with open('./vis_eb_train_sorted.pkl', 'wb') as f:
			pickle.dump(recall, f, pickle.HIGHEST_PROTOCOL)
	# DrawLib.drawBar(range(length),"proposal number",
	# 				recall, "recall",
	# 				)
	# ss: unique+filter small 88.24118589743589% / 2759
	#### ss: test  91.19257478632479% / 3760
	#### ss: train 90.78661729025668% / 3879
	#### ss: all   90.98505124355376% / 3879
	# eb: >0.1 73.02%/ 1200
	# eb: 89.9% / 5000
	#### eb: train 93.45549738219895% / 9135
	#### eb: train 91.192% / 5503

# mat2xml(matname='/home/disk/dataset/SelectiveSearchVOC2007trainval.mat', # test.mat
# 		img_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/JPEGImages',
# 		anno_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/SSAnnotation')

# edgebox2xml(
# 		filename='/home/disk/dataset/EdgeBoxes70.h5',
# 		img_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/JPEGImages',
# 		anno_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/EBAnnotation')


def merge(input1,input2,output):
	if not os.path.exists(output):
		os.mkdir(output)
	filelist = os.listdir(input1)
	for anno_name in tqdm(filelist):
		annopath1 = os.path.join(input1, anno_name)
		annopath2 = os.path.join(input2, anno_name)
		outpath = os.path.join(output, anno_name)
		if not os.path.exists(annopath1) or not os.path.exists(annopath2):continue
		objs1 = parse_rec(annopath1)
		objs2 = parse_rec(annopath2)
		writer = PascalVocWriter("", "", [500,500,3])
		for obj1 in objs1:
			ind, miou = None,0
			for obj2 in objs2:
				_iou = iou(obj1['bbox'], obj2['bbox'])
				if _iou>miou:
					miou = _iou
					ind = obj2['bbox']
			print (miou, anno_name)
			x1,y1,x2,y2 = ind
			writer.addBndBox(x1,y1,x2,y2,obj1['name'],-1,obj1['score'])
		writer.save(outpath)

	pass

if __name__ == '__main__':
	cal_recall()
	# merge('/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/CAMAnnotation',
	# 	  '/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/SSAnnotation',
	# 	  '/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/CAMSSAnnotation')


	# mat2xml(matname='/home/disk/dataset/SelectiveSearchVOC2007trainval.mat', # test.mat
	# 	img_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/JPEGImages',
	# 	anno_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/SSAnnotation')

	# edgebox2xml(
	# 		filename='/home/disk/dataset/EdgeBoxes70.h5',
	# 		img_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/JPEGImages',
	# 		anno_dir='/home/disk/dataset/voc/VOC/VOCdevkit2007/VOC2007/tmpEBAnnotation')