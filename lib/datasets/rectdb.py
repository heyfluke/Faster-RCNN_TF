# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
import json
from fast_rcnn.config import cfg
import pdb


class rectdb(imdb):
    def __init__(self, image_set, folder_name='rectdb', classes=None):
        imdb.__init__(self, folder_name + '_' + image_set)
        self._folder_name = folder_name
        self._image_set = image_set
        self._data_path = self._get_default_path() #if data_path is None else data_path
        if classes and isinstance(classes, list) and len(classes):
            self._classes = tuple(['__background__'] + classes)
        else:
            self._classes = ('__background__', # always index 0
                         'text')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        print('self._class_to_ind', self._class_to_ind)
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'min_size'    : 2,
                       'top_k'       : 2000}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        self._annotation_data = None

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, self._folder_name)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _preload_annotation(self):
        filename = os.path.join(self._data_path, 'ground_truth.json')
        print('annotation file', filename)
        o = json.load(open(filename, 'r'))
        self._annotation_data = {}
        for item in o:
            self._annotation_data[item['filename']] = item
        print('len _annotation_data', len(self._annotation_data))

    def _load_annotation(self, index):
        if not self._annotation_data:
            self._preload_annotation()
        o = self._annotation_data[index]
        num_objs = len(o['rects'])

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, rect_info in enumerate(o['rects']):
            rect, label = rect_info[0], rect_info[1]
            cls = self._class_to_ind[label]
            # boxes[ix, :] = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]
            boxes[ix, :] = [rect[0], rect[1], rect[2]+rect[0]-1, rect[3]+rect[1]-1]
            if boxes[ix][2] <= boxes[ix][0]:
                print(boxes[ix])
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.rectdb import rectdb
    d = rectdb('train')
    res = d.roidb
    from IPython import embed; embed()
