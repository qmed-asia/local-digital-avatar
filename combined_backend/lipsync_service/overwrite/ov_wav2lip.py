# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import os

#script_folder = os.path.join(os.path.dirname(__file__), 'wav2lip')
#sys.path.append(script_folder)

import numpy as np
import cv2
from ..audio import load_wav, melspectrogram
import subprocess
from tqdm import tqdm
from glob import glob
import torch
import platform
import openvino as ov
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import torch.nn.functional as F
import openvino.properties.hint as hints
import openvino.properties as props
# from face_parsing import init_parser, swap_regions
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import combined_backend.config as config

import time
import logging
import uuid
from pathlib import Path

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class OVFaceDetector(object):
    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose

    def detect_from_image(self, tensor_or_path):
        raise NotImplementedError

    def detect_from_directory(self, path, extensions=['.jpg', '.png'], recursive=False, show_progress_bar=True):
        if self.verbose:
            logger = logging.getLogger(__name__)

        if len(extensions) == 0:
            if self.verbose:
                logger.error(
                    "Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = '/**/*' if recursive else '/*'
        files = []
        for extension in extensions:
            files.extend(glob.glob(path + additional_pattern +
                         extension, recursive=recursive))

        if self.verbose:
            logger.info(
                "Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}
        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info(
                    "Running the face detector on image: %s", image_path)
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info(
                "The detector was successfully run on all %s images", len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else cv2.imread(tensor_or_path)[..., ::-1]
        elif torch.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError


class OVSFDDetector(OVFaceDetector):
    def __init__(self, device, face_detector, verbose=False):
        super().__init__(device, verbose)
        self.face_detector = face_detector

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = self.detect(self.face_detector, image, device="cpu")
        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = self.batch_detect(self.face_detector, images, device="cpu")
        keeps = [self.nms(bboxlists[:, i, :], 0.3)
                 for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5]
                     for bboxlist in bboxlists]

        return bboxlists

    def nms(self, dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:,
                                                  1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(
                y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(
                y2[i], y2[order[1:]])

            w, h = np.maximum(
                0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(self, net, img, device):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)

        img = torch.from_numpy(img).float().to(device)
        BB, CC, HH, WW = img.size()

        results = net({"x": img.numpy()})
        olist = [torch.Tensor(results[i]) for i in range(12)]

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        return bboxlist

    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def batch_detect(self, net, imgs, device):
        imgs = imgs - np.array([104, 117, 123])
        imgs = imgs.transpose(0, 3, 1, 2)

        imgs = torch.from_numpy(imgs).float().to(device)
        BB, CC, HH, WW = imgs.size()

        results = net({"x": imgs.numpy()})
        olist = [torch.Tensor(results[i]) for i in range(12)]
        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
            # olist[i * 2] = (olist[i * 2], dim=1)
        olist = [oelem.data.cpu() for oelem in olist]
        # olist = [oelem for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[:, 1, hindex, windex]
                loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)
                priors = torch.Tensor(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
                variances = [0.1, 0.2]
                box = self.batch_decode(loc, priors, variances)
                box = box[:, 0] * 1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append(
                    torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, BB, 5))

        return bboxlist

    def batch_decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class OVFaceAlignment:
    def __init__(self, landmarks_type, face_detector, network_size=NetworkSize.LARGE,
                 device='CPU', flip_input=False, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        self.face_detector = OVSFDDetector(
            device=device, face_detector=face_detector, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results


class OVWav2Lip:
    def __init__(self, compiled_face_detector, compiled_wav2lip_model, enhancer, full_frames, face_det_results, fps, static, duration=None):
        # Store initialized components
        self.compiled_face_detector = compiled_face_detector # Store the compiled face detector
        self.compiled_wav2lip_model = compiled_wav2lip_model # Store the compiled Wav2Lip model
        self.enhancer = enhancer # Store the enhancer instance
        self.lock = Lock() # Keep the lock if using ThreadPoolExecutor with enhancer

        # Store pre-processed avatar data and info
        self.full_frames = full_frames # Store the full list of avatar frames
        self.face_det_results = face_det_results # Store the face detection results for avatar frames
        self.fps = fps # Store the video FPS
        self.duration = duration # Store video duration (if applicable)
        self.static = static

        # Video processing settings
        # self.static = False # This should be determined based on full_frames or passed in
        self.pads = [0, 15, -10, -10]
        self.face_det_batch_size = 8 
        self.wav2lip_batch_size = 128
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1] 
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = True 
        self.with_face_mask = False 
        self.no_segmentation = False 
        self.no_sr = False 
        self.img_size = 96 

        # Audio processing settings
        self.mel_step_size = 16

        # Model URLs
        self.models_urls = {
            's3fd': '[https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth)',
        }



    def process_images_with_detector(self, images, detector, initial_batch_size): # Called by face_detect_ov()
        print("DEBUG: Entering process_images_with_detector method.", flush=True)
        batch_size = initial_batch_size
        predictions = []

        for _ in range(int(np.log2(initial_batch_size)) + 1):  # Loop to halve the batch size
            try:
                print(f"DEBUG: Attempting processing with batch size: {batch_size}", flush=True) # Add print
                # Use enumerate with tqdm to track progress and index
                for i, batch_start_index in enumerate(range(0, len(images), batch_size)):
                    #print(f"DEBUG: Processing batch {i} starting at index {batch_start_index}", flush=True)
                    batch_images = np.array(images[batch_start_index:batch_start_index + batch_size])
                    #print("DEBUG: Calling detector.get_detections_for_batch(batch_images)...", flush=True)
                    batch_predictions = detector.get_detections_for_batch(batch_images)
                    #print("DEBUG: detector.get_detections_for_batch(batch_images) complete.", flush=True)
                    predictions.extend(batch_predictions)
                    #print(f"DEBUG: Processed batch {i} starting at index {batch_start_index}.", flush=True)

                #print("DEBUG: Exiting process_images_with_detector loop successfully.", flush=True)
                return predictions
            except RuntimeError as e:
                print(f"ERROR: RuntimeError in process_images_with_detector with batch size {batch_size}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
                
                

    def face_detect_ov(self, images, device):
        print("DEBUG: Entering face_detect_ov method.", flush=True)
        print("DEBUG: Initializing OVFaceAlignment...", flush=True)
        try:
            detector = OVFaceAlignment(
                LandmarksType._2D, face_detector=self.compiled_face_detector, flip_input=False, device=device)
            #print("DEBUG: OVFaceAlignment initialized.", flush=True)

            #print(f"DEBUG: Processing {len(images)} images with detector...", flush=True) 
            predictions = self.process_images_with_detector(images, detector, self.face_det_batch_size)
            #print("DEBUG: Image processing with detector complete.", flush=True)

            results = [] # Based on the original code structure in run() using face_det_results
            pady1, pady2, padx1, padx2 = self.pads

            #print("DEBUG: Processing face detection results...", flush=True)

            for rect, image in zip(predictions, images):
                if rect is None:
                    cv2.imwrite('temp/faulty_frame.jpg', image)
                    print(f"ERROR: Face not detected in a frame: {len(results)}", flush=True)
                    raise ValueError(
                        'Face not detected! Ensure the video contains a face in all the frames.')

                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)

                results.append([image[y1: y2, x1:x2], (y1, y2, x1, x2)])

            boxes = np.array([r[1] for r in results])
            if not self.nosmooth:
                 print("DEBUG: Applying box smoothing.", flush=True)
                 boxes = self.get_smoothened_boxes(boxes, T=5)
                 # Update results with smoothened boxes
                 for i, (img_crop, original_box) in enumerate(results):
                     results[i] = [img_crop, boxes[i].tolist()]

            #print("DEBUG: Face detection results processed, exiting face_detect_ov.", flush=True)
            return results

        except Exception as e:
            print(f"ERROR: Exception during face_detect_ov: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise # Re-raise the exception
        
    def get_smoothened_boxes(boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_full_frames_and_face_det_results(self, reverse=False, double=False):
        if reverse:
            full_frames = self.full_frames.copy()[::-1]
            face_det_results = self.face_det_results.copy()[::-1]
        else:
            full_frames = self.full_frames.copy()
            face_det_results = self.face_det_results.copy()
        
        if double:
            full_frames = full_frames + full_frames[::-1]
            face_det_results = face_det_results + face_det_results[::-1]
        
        return full_frames, face_det_results

    def inference(self, audio_path, reversed=False, starting_frame=0, enhance=False, output_dir=None):
        # Keep the inner process_frame function as is
        def process_frame(index, p, f, c, enhancer, enhance):
            y1, y2, x1, x2 = c
            p = p.astype(np.uint8)

            # Adjust coords to only focus on mouth
            crop_coords = [20, 80, 60, 90] # x1, x2, x3, x4

            width_c = x2 - x1
            height_c = y2 - y1
            # Check if p.shape[1] or p.shape[0] are zero or negative before division
            scale_x = width_c / p.shape[1] if p.shape[1] > 0 else 0
            scale_y = height_c / p.shape[0] if p.shape[0] > 0 else 0

            p = p[crop_coords[2]: crop_coords[3], crop_coords[0]: crop_coords[1]]

            # Adjust the crop_coords relative to c
            adjusted_crop_coords = [
                int(crop_coords[0] * scale_x),
                int(crop_coords[1] * scale_x),
                int(crop_coords[2] * scale_y),
                int(crop_coords[3] * scale_y)
            ]
            x1 = x1 + adjusted_crop_coords[0]
            x2 = x1 + (adjusted_crop_coords[1] - adjusted_crop_coords[0])
            y1 = y1 + adjusted_crop_coords[2]
            y2 = y1 + (adjusted_crop_coords[3] - adjusted_crop_coords[2])

            if enhancer and enhance:
                with self.lock:
                    try:
                        p, _ = enhancer.enhance(p) 
                    except Exception as e:
                        print(f"WARNING: Enhancer failed for a frame: {e}", flush=True)
                        pass


            # Check if resized dimensions are valid
            resized_w = int((crop_coords[1] - crop_coords[0]) * scale_x)
            resized_h = int((crop_coords[3] - crop_coords[2]) * scale_y)
            if resized_w <= 0 or resized_h <= 0:
                 #print(f"WARNING: Invalid dimensions for cv2.resize: ({resized_w}, {resized_h}) at index {index}", flush=True)
                 return index, f # Skip processing this frame, return original

            p = cv2.resize(p, (resized_w, resized_h))

            # Create a mask for seamless cloning
            mask = 255 * np.ones(p.shape[:2], dtype=p.dtype) # Mask is 2D (height, width)
            
            # Calculate the center of the region where the image will be cloned
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # Check if the center and region are valid for seamlessClone
            # seamlessClone can be sensitive to boundary conditions and shape mismatches
            try:
                # Perform seamless cloning
                # Ensure source (p) and destination (f) have the same number of channels
                if p.shape[-1] != f.shape[-1]:
                     #print(f"WARNING: Channel mismatch for seamlessClone at index {index}: source {p.shape}, dest {f.shape}", flush=True)
                     # Convert p to match f's channels if needed (e.g., grayscale to BGR)
                     if p.ndim == 2 and f.ndim == 3 and f.shape[-1] == 3:
                         p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
                     elif p.ndim == 3 and f.ndim == 2 and p.shape[-1] == 3:
                          print(f"WARNING: Cannot convert BGR source to grayscale dest for seamlessClone at index {index}.", flush=True)
                          return index, f

                # Ensure the mask matches the spatial dimensions of the source (p)
                if mask.shape[:2] != p.shape[:2]:
                     #print(f"WARNING: Mask shape mismatch for seamlessClone at index {index}: mask {mask.shape}, source {p.shape}", flush=True)
                     return index, f # Skip cloning

                f = cv2.seamlessClone(p, f, mask, center, cv2.NORMAL_CLONE)
            except cv2.error as e:
                 print(f"WARNING: OpenCV Error during seamlessClone at index {index}: {e}", flush=True)
                 return index, f
            except Exception as e:
                 print(f"WARNING: Unexpected error during seamlessClone at index {index}: {e}", flush=True)
                 return index, f

            return index, f

        file_id = str(uuid.uuid4())

        # Construct the final output path using output_dir
        if output_dir:
            output_path = os.path.join(output_dir, f"{file_id}.mp4")
            print(f"DEBUG: [Inference Method] Saving final MP4 result to configured directory: {output_path}", flush=True)
        else:
            fallback_dir = os.path.join(os.getcwd(), 'lipsync_results_fallback') # Fallback to a dir in current working dir
            os.makedirs(fallback_dir, exist_ok=True)
            output_path = os.path.join(fallback_dir, f"{file_id}.mp4")
            print(f"WARNING: [Inference Method] output_dir not provided. Saving to fallback directory: {output_path}", flush=True)


        # Ensure the final output directory exists (important for fallback too)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        wav, wav_duration = load_wav(audio_path, 16000)
        mel = melspectrogram(wav)

        mel_chunks = []
        new_frames = []
        gen = None
        mel_idx_multiplier = 80./self.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(
                mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        if not self.static:
            full_frames, face_det_results = self.get_full_frames_and_face_det_results(reverse=reversed, double=True)
        else:
            full_frames, face_det_results = self.get_full_frames_and_face_det_results()

        # --- Create temporary directory and file path for the AVI ---
        # Use config.DATA_DIRECTORY for a reliable temporary location
        temp_dir = os.path.join(config.DATA_DIRECTORY, 'lipsync-temp')
        os.makedirs(temp_dir, exist_ok=True)

        temp_avi_filename = f'{file_id}_temp_result.avi'
        temp_avi_path = os.path.join(temp_dir, temp_avi_filename)

        print(f"DEBUG: [Inference Method] Using temporary AVI path: {temp_avi_path}", flush=True) # Added log


        gen = self.datagen(full_frames.copy(), mel_chunks, face_det_results, start_index=starting_frame)
        new_frames = full_frames.copy()[:len(mel_chunks)]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        frames_generated = 0

        # --- VideoWriter Initialization ---
        frame_h, frame_w = new_frames[0].shape[:-1]
        # Initialize cv2.VideoWriter with the NEW temporary AVI path
        fourcc = cv2.VideoWriter_fourcc(*'DIVX') # Ensure DIVX codec is available (often is)
        # Consider using 'MJPG' or 'XVID' if DIVX causes issues on your system
        out = cv2.VideoWriter(temp_avi_path, fourcc, self.fps, (frame_w, frame_h))

        if not out.isOpened(): # Check if VideoWriter was opened successfully
             # Attempt cleanup of the file created by VideoWriter if it exists
             if os.path.exists(temp_avi_path):
                  try: os.remove(temp_avi_path)
                  except: pass
             raise RuntimeError(f"cv2.VideoWriter failed to open {temp_avi_path}. Check codec (DIVX) and file path.")


        # --- Frame Processing and Writing Loop ---
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(np.ceil(float(len(mel_chunks))/self.face_det_batch_size)))):
            if i == 0:
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                pred_ov = self.compiled_wav2lip_model({"audio_sequences": mel_batch.numpy(), "face_sequences": img_batch.numpy()})[0]
            else:
                img_batch = np.transpose(img_batch, (0, 3, 1, 2))
                mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
                pred_ov = self.compiled_wav2lip_model({"audio_sequences": mel_batch, "face_sequences": img_batch})[0]

            pred_ov = pred_ov.transpose(0, 2, 3, 1) * 255.

            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_frame, i, p, f, c, self.enhancer, enhance)
                           for i, (p, f, c) in enumerate(zip(pred_ov, frames, coords))]
                results = [None] * len(futures) # Pre-allocate results list
                for future in futures:
                    # This waits for each future to complete
                    try:
                        index, processed_frame = future.result()
                        results[index] = processed_frame
                        # print(f"DEBUG: [Inference Method] Processed frame {index}.", flush=True) # Verbose log
                    except Exception as e:
                         print(f"ERROR: [Inference Method] Error processing frame {index}: {e}", flush=True)
                         results[index] = frames[index]
                         pass

            for f in results:
                if f is not None: # Check if frame processing returned a valid frame
                     frames_generated += 1
                     out.write(f)
                else:
                     print(f"WARNING: [Inference Method] Skipping writing a None frame.", flush=True)

        # --- Release VideoWriter ---
        out.release()

        # --- Check if temporary AVI file exists ---
        if os.path.exists(temp_avi_path):
            print()
        else:
             print(f"ERROR: [Inference Method] CONFIRMATION: Temporary AVI file NOT FOUND at {temp_avi_path} after VideoWriter release.", flush=True) # Added log
             raise RuntimeError(f"Failed to create temporary AVI file: {temp_avi_path}. Check VideoWriter initialization.")


        # --- FFmpeg Command ---

        command = [
            'ffmpeg', '-y',
            '-i', audio_path,       # Input audio
            '-i', temp_avi_path,    # Input video (temporary AVI)
            '-strict', '2',
            '-q:v', '1',            # Output video quality (1 is highest for libx264, check for other codecs)
            '-vf', "hqdn3d,unsharp=5:5:0.5", # Video filters
            output_path             # Final output MP4 path
        ]


        # --- Execute FFmpeg subprocess ---
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False) # check=False to not raise exception automatically

            # Check FFmpeg return code
            if process.returncode != 0:
                 print(f"ERROR: [Inference Method] FFmpeg command failed with return code {process.returncode}", flush=True)
                 raise RuntimeError(f"FFmpeg command failed with return code {process.returncode}. See server logs for STDERR/STDOUT.")


        except FileNotFoundError:
             print(f"ERROR: [Inference Method] FFmpeg executable not found. Is FFmpeg installed and in PATH?", flush=True) # Added log
             raise RuntimeError("FFmpeg executable not found. Please ensure FFmpeg is installed and accessible in your system's PATH.")
        except Exception as e:
             print(f"ERROR: [Inference Method] Error running FFmpeg subprocess: {e}", flush=True) # Added log
             # Consider raising an error if FFmpeg fails
             raise RuntimeError(f"FFmpeg subprocess error: {e}") from e


        if os.path.exists(output_path):
            print() # Added log
        else:
             print(f"ERROR: [Inference Method] CONFIRMATION: Final MP4 file NOT FOUND at {output_path} after FFmpeg.", flush=True) # Added log
             # Raise an error here if the final file is missing
             raise RuntimeError(f"FFmpeg failed to create final output file: {output_path}. Check FFmpeg logs above for details.")

        try:
            os.remove(temp_avi_path)
        except Exception as cleanup_e:
            print(f"WARNING: [Inference Method] Failed to clean up temporary AVI file {temp_avi_path}: {cleanup_e}", flush=True) # Added log


        # The method returns the file_id string
        return file_id

    def datagen(self, frames, mels, face_det_results, start_index=0):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        num_frames = len(frames)
        
        for i, m in enumerate(mels):
            # Start from the specified index, wrapping around the frame list if necessary
            idx = (start_index + i) % num_frames if not self.static else 0
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate(
                    (img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

if __name__ == "__main__":
    init_time = time.time()
    wav2lip = OVWav2Lip(device="CPU", avatar_path="assets/image.png")
    print("Init Time: ", time.time() - init_time)
    start_time = time.time()
    wav2lip.inference("assets/audio.wav")
    print("Time taken: ", time.time() - start_time)
