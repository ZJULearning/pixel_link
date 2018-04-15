#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util;
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example

# encoding = utf-8
import numpy as np    
import time
import config
import util  


class SynthTextDataFetcher():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()
        
    @util.dec.print_calling    
    def _load_mat(self):
        data = util.io.load_mat(self.mat_path)
        self.image_paths = data['imnames'][0]
        self.image_bbox = data['wordBB'][0]
        self.txts = data['txt'][0]
        self.num_images =  len(self.image_paths)

    def get_image_path(self, idx):
        image_path = util.io.join_path(self.root_path, self.image_paths[idx][0])
        return image_path

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except: # error caused by dataset
            return 1


    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) ==2: # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))
             
        xys = boxes[:,:, word_idx]
        assert(np.shape(xys) ==(2, 4))
        return np.float32(xys)
    
    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        
        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)
        
        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True
        
        if w < 10 or h < 10:
            is_valid = False
            
        if w * h < 100:
            is_valid = False
        
        xys[0, :] = xys[0, :] / width
        xys[1, :] = xys[1, :] / height
        
        return is_valid, min_x / width, min_y /height, max_x / width, max_y / height, xys
        
    def get_txt(self, image_idx, word_idx):
        txts = self.txts[image_idx];
        clean_txts = []
        for txt in txts:
            clean_txts += txt.split()
        return str(clean_txts[word_idx])
        
        
    def fetch_record(self, image_idx):
        image_path = self.get_image_path(image_idx)
        if not (util.io.exists(image_path)):
            return None;
        img = util.img.imread(image_path)
        h, w = img.shape[0:-1];
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        full_bboxes = []
        txts = []
        for word_idx in xrange(num_words):
            xys = self.get_word_bbox(image_idx, word_idx);       
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width = w, height = h)
            if not is_valid:
                continue;
            rect_bboxes.append([min_x, min_y, max_x, max_y])
            xys = np.reshape(np.transpose(xys), -1)
            full_bboxes.append(xys);
            txt = self.get_txt(image_idx, word_idx);
            txts.append(txt);
        if len(rect_bboxes) == 0:
            return None;
        
        return image_path, img, txts, rect_bboxes, full_bboxes
    
        

def cvt_to_tfrecords(output_path , data_path, gt_path, records_per_file = 50000):

    fetcher = SynthTextDataFetcher(root_path = data_path, mat_path = gt_path)
    image_idxes = range(fetcher.num_images)
    np.random.shuffle(image_idxes)
    record_count = 0;
    for image_idx in image_idxes:
        if record_count % records_per_file == 0:
            fid = record_count / records_per_file
            tfrecord_writer = tf.python_io.TFRecordWriter(output_path%(fid))

        print "converting image %d/%d"%(record_count, fetcher.num_images)
        record = fetcher.fetch_record(image_idx);
        if record is None:
            print '\nimage %d does not exist'%(image_idx + 1)
            continue;
        record_count += 1
        image_path, image, txts, rect_bboxes, oriented_bboxes = record;
        labels = [];
        for txt in txts:
            if len(txt) < 3:
                labels.append(config.ignore_label)
            else:
                labels.append(config.text_label)
        image_data = tf.gfile.FastGFile(image_path, 'r').read()
        shape = image.shape
        image_name = str(util.io.get_filename(image_path).split('.')[0])
        example = convert_to_example(image_data, image_name, labels, txts, rect_bboxes, oriented_bboxes, shape)
        tfrecord_writer.write(example.SerializeToString())
                
                    
if __name__ == "__main__":
    mat_path = util.io.get_absolute_path('~/dataset/SynthText/gt.mat')
    root_path = util.io.get_absolute_path('~/dataset/SynthText/')
    output_dir = util.io.get_absolute_path('~/dataset/pixel_link/SynthText/')
    util.io.mkdir(output_dir);
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir,  'SynthText_%d.tfrecord'), data_path = root_path, gt_path = mat_path)
