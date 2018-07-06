import numpy as np
from random import shuffle
import imageio
import zipfile
import tarfile
import os
import sys
import urllib.request

from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import array_to_img
from PIL import Image

class Scene_Flow_disparity(object):
    '''
    Scene Flow Datasets
    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    '''
    def __init__(self, dataset_path = "./data/Scene Flow Datasets/"):
        self.height = 540
        self.width = 960
        self.channels = 3

        self.dataset_path = dataset_path
        download_url = [
            # RGB images (cleanpass) png files
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_cleanpass.tar',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar',
            # RGB images (finalpass) png files
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar',
            # Disparity pfm ground truth
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2',
            'https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2'
        ]

        self.data_paths = []
        for url in download_url:
            self.download_and_extract(url)
            filename = url.split('/')[-1]
            self.check_data_dir_path(filename)
        
        shuffle(self.data_paths)
        print("complete loading Scene Flow Datasets")

    def download_and_extract(self, url):

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        filename = url.split('/')[-1]
        filepath = os.path.join(self.dataset_path, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('\r>> Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        else:
            print("\r>> already download", filename,'of Scene Flow Datasets')
        if not os.path.exists(self.dataset_path + '/' + filename.split('.')[0]):
            if filename.split('.')[-1] == "bz2":
                tar = tarfile.open(self.dataset_path + filename, "r:bz2")
            else:
                tar = tarfile.open(self.dataset_path + filename, "r:tar")
            tar.extractall(self.dataset_path + '/' + filename.split('.')[0])
            tar.close()
            print('\r>> Successfully extracted', filename)
        else:
            print('\r>> already extracted', filename.split('.')[0], 'of Scene Flow Datasets')

    def check_data_dir_path(self, filename):

        for (path, dir, files) in os.walk(self.dataset_path + '/' + filename.split('.')[0] + '/'):
            for filename_ in files:
                self.data_paths.append(path.replace("\\", "/") + '/' + filename_)

    def _get_next_line(self, f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    def write_pfm(self, data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
        # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

        data = np.flipud(data)
        height, width = np.shape(data)[:2]
        values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
        endianess = data.dtype.byteorder
        # print(endianess)

        if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
            scale *= -1

        with open(fpath, 'wb') as file:
            file.write((file_identifier))
            file.write(('\n%d %d\n' % (width, height)).encode())
            file.write(('%d\n' % scale).encode())

            file.write(values)
            
    def read_pfm(self, fpath, expected_identifier="Pf"):
        # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

        with open(fpath, 'rb') as f:
            #  header
            identifier = self._get_next_line(f)
            if identifier != expected_identifier:
                raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

            try:
                line_dimensions = self._get_next_line(f)
                dimensions = line_dimensions.split(' ')
                width = int(dimensions[0].strip())
                height = int(dimensions[1].strip())
            except:
                raise Exception('Could not parse dimensions: "%s". '
                                'Expected "width height", e.g. "512 512".' % line_dimensions)

            try:
                line_scale = self._get_next_line(f)
                scale = float(line_scale)
                assert scale != 0
                if scale < 0:
                    endianness = "<"
                else:
                    endianness = ">"
            except:
                raise Exception('Could not parse max value / endianess information: "%s". '
                                'Should be a non-zero number.' % line_scale)

            try:
                data = np.fromfile(f, "%sf" % endianness)
                data = np.reshape(data, (height, width))
                data = np.flipud(data)
                with np.errstate(invalid="ignore"):
                    data *= abs(scale)
            except:
                raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

            return data 

    def groundTruth(self, image_file_path, mode = 'disparity'):
        
        tmp = image_file_path.replace(".png", ".pfm")

        if mode == 'disparity':
            if 'driving__frames_cleanpass' in image_file_path:
                tmp1 = tmp.replace("driving__frames_cleanpass", "driving__disparity")
                tmp = tmp1.replace("frames_cleanpass", "disparity")
            elif 'driving__frames_finalpass' in image_file_path:
                tmp1 = tmp.replace("driving__frames_finalpass", "driving__disparity")
                tmp = tmp1.replace("frames_finalpass", "disparity")
            elif 'flyingthings3d__frames_cleanpass' in image_file_path:
                tmp1 = tmp.replace("flyingthings3d__frames_cleanpass", "flyingthings3d__disparity")
                tmp = tmp1.replace("frames_cleanpass", "disparity")
            elif 'flyingthings3d__frames_finalpass' in image_file_path:
                tmp1 = tmp.replace("flyingthings3d__frames_finalpass", "flyingthings3d__disparity")
                tmp = tmp1.replace("frames_finalpass", "disparity")
            elif 'monkaa__frames_cleanpass' in image_file_path:
                tmp1 = tmp.replace("monkaa__frames_cleanpass", "monkaa__disparity")
                tmp = tmp1.replace("frames_cleanpass", "disparity")
            elif 'monkaa__frames_finalpass' in image_file_path:
                tmp1 = tmp.replace("monkaa__frames_finalpass", "monkaa__disparity")
                tmp = tmp1.replace("frames_finalpass", "disparity")

            groundTruth = self.read_pfm(fpath = tmp)
            return groundTruth

    def data(self, image_file_path, image_size = (540, 960), truth_size = (540, 960), mode = 'left', T_mode = 'disparity'):

        if mode == 'left':
            if '.png' in image_file_path:
                if 'left' in image_file_path:
                    l_img = load_img(path = image_file_path, grayscale = False, target_size = image_size, interpolation = 'bicubic')
                    r_img = load_img(path = image_file_path.replace("left", "right"), grayscale = False, target_size = image_size, interpolation = 'bicubic')

                    left_image = img_to_array(l_img)
                    right_image = img_to_array(r_img)
                    
                    ground_truth = array_to_img(self.groundTruth(image_file_path, T_mode)[:,:,np.newaxis])
                    resized_truth = ground_truth.resize((truth_size[-1], truth_size[0]))
                    ground_truth = img_to_array(resized_truth)

                    return left_image, right_image, ground_truth
                else:
                    # print('no left path')
                    return
            else:  
                # print('no png path')
                return    

class Light_Field_Dataset(object):
    '''
    4D Light Field Dataset
    http://hci-lightfield.iwr.uni-heidelberg.de/
    '''
    def __init__(self, dataset_path = "./data/4D Light Field Dataset/"):
        self.height = 512
        self.width = 512
        self.channels = 3

        self.dataset_path = dataset_path
        self.dataset_url = "http://lightfield-analysis.net/benchmark/downloads/full_data.zip"

        self.download_and_extract()
        
        self.image_paths = []
        self.check_data_dir_path()
        print("make light_field dictionary by image_paths indexing")

    def download_and_extract(self):

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        filename = self.dataset_url.split('/')[-1]
        filepath = os.path.join(self.dataset_path, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self.dataset_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        else:
            print("already download 4D Light Field Dataset")
        if not os.path.exists(self.dataset_path + '/full_data'):
            zipfile.ZipFile(filepath).extractall(self.dataset_path)
            zipfile.ZipFile(filepath).close()
            print('Successfully extracted', filename, statinfo.st_size, 'bytes.')
        else:
            print("already extracted 4D Light Field Dataset")

    def check_data_dir_path(self):

        for (path, dir, files) in os.walk(self.dataset_path + 'full_data/'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.cfg':
                    self.image_paths.append(path.replace("\\", "/"))

    def _get_next_line(self, f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    def write_pfm(self, data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
        # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

        data = np.flipud(data)
        height, width = np.shape(data)[:2]
        values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
        endianess = data.dtype.byteorder
        # print(endianess)

        if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
            scale *= -1

        with open(fpath, 'wb') as file:
            file.write((file_identifier))
            file.write(('\n%d %d\n' % (width, height)).encode())
            file.write(('%d\n' % scale).encode())

            file.write(values)
            
    def read_pfm(self, fpath, expected_identifier="Pf"):
        # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

        with open(fpath, 'rb') as f:
            #  header
            identifier = self._get_next_line(f)
            if identifier != expected_identifier:
                raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

            try:
                line_dimensions = self._get_next_line(f)
                dimensions = line_dimensions.split(' ')
                width = int(dimensions[0].strip())
                height = int(dimensions[1].strip())
            except:
                raise Exception('Could not parse dimensions: "%s". '
                                'Expected "width height", e.g. "512 512".' % line_dimensions)

            try:
                line_scale = self._get_next_line(f)
                scale = float(line_scale)
                assert scale != 0
                if scale < 0:
                    endianness = "<"
                else:
                    endianness = ">"
            except:
                raise Exception('Could not parse max value / endianess information: "%s". '
                                'Should be a non-zero number.' % line_scale)

            try:
                data = np.fromfile(f, "%sf" % endianness)
                data = np.reshape(data, (height, width))
                data = np.flipud(data)
                with np.errstate(invalid="ignore"):
                    data *= abs(scale)
            except:
                raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

            return data 

    def data(self, image_path):
        
        images = []

        for i in range(0, 81): 
            image_file_path = image_path + '/input_Cam0%.2d.png' % i
            img = load_img(path = image_file_path, grayscale = True, interpolation = 'bicubic')
            images.append(img_to_array(img))

        return images
    
    def groundTruth(self, image_path):

        if 'test' in image_path:
            print('there are not ground truth')
            return 0
            
        groundTruth_path = image_path + '/gt_disp_lowres.pfm'
        groundTruth = self.read_pfm(fpath = groundTruth_path)
        return groundTruth
    