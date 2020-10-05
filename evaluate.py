# from matplotlib import pyplot as plt
import json
import os
import shutil

import cv2
import numpy as np
import tvm
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from tvm.contrib import graph_runtime
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from core.yolov4 import decode, filter_boxes
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')



def my_decode(feature_maps):
    global NUM_CLASS, STRIDES, ANCHORS, XYSCALE

    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
      with tf.name_scope("featuremap-"+str(i)) as scope:
        if i == 0:
          output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        elif i == 1:
          output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
          output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    pred = (pred_bbox, pred_prob)
    return pred


def my_dequantize(tensor, scale, zp):
    return np.multiply(scale, (tensor.astype(np.float32) - zp))

def main(_argv):
    global NUM_CLASS, STRIDES, ANCHORS, XYSCALE

    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    elif  FLAGS.framework == 'tvm':
        ctx = tvm.cpu(0)
        loaded_graph = open(os.path.join(FLAGS.weights, "modelDescription.json")).read()
        loaded_lib = tvm.runtime.load_module(os.path.join(FLAGS.weights, "modelLibrary.so"))
        loaded_params = bytearray(open(os.path.join(FLAGS.weights,  "modelParams.params"), "rb").read())
        #
        # Get rid of the leip key
        #
        graphjson = json.loads(loaded_graph)
        if 'leip' in list(graphjson.keys()):
            del graphjson['leip']
            loaded_graph = json.dumps(graphjson)

        m = graph_runtime.create(loaded_graph, loaded_lib, ctx)
        m.load_params(loaded_params)
    #elif ('.h5' in FLAGS.weights):
    #    saved_model_loaded = tf.keras.models.load_model(FLAGS.weights)
    #    infer = saved_model_loaded.predict
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        print(saved_model_loaded.signatures)

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))

            if FLAGS.framework == 'tflite':
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                image_data_casted = image_data.astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], image_data_casted)
                interpreter.invoke()
                # pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                #     boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
                # else:
                #     boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
                fm1 = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
                fm2 = interpreter.get_tensor(output_details[1]['index']).astype(np.float32)
                fm3 = interpreter.get_tensor(output_details[2]['index']).astype(np.float32)
                print(fm1.shape)
                print(fm2.shape)
                print(fm3.shape)
                fm1 = my_dequantize(fm1.astype(np.float32), 1.1345850229263306, 223)
                fm2 = my_dequantize(fm2.astype(np.float32), 2.054811954498291, 242)
                fm3 = my_dequantize(fm3.astype(np.float32), 8.428282737731934, 248)
                pred = my_decode([fm1, fm2, fm3]) # these need to be ordered biggest tensor to smallest I think
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=FLAGS.score)

            elif FLAGS.framework == 'tvm':
                # image_data = image_data / 255. # DO NOT DIVIDE by 255 for uint8 eval!
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                image_data_casted = image_data.astype(np.uint8)
                m.set_input("input_1", tvm.nd.array(image_data_casted))
                ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=1)
                prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
                fm1 = m.get_output(0).asnumpy()
                fm2 = m.get_output(1).asnumpy()
                fm3 = m.get_output(2).asnumpy()
                print(fm1.shape)
                print(fm2.shape)
                print(fm3.shape)
                fm1 = my_dequantize(fm1.astype(np.float32), 1.1345850229263306, 223)
                fm2 = my_dequantize(fm2.astype(np.float32), 2.054811954498291, 242)
                fm3 = my_dequantize(fm3.astype(np.float32), 8.428282737731934, 248)

                pred = my_decode([fm1, fm2, fm3]) # these need to be ordered biggest tensor to smallest I think
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=FLAGS.score)

                #exit()
            else:
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                #print(pred_bbox)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

                print("boxes = ", boxes)
                print("predictions = ", pred_conf)
 
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            print(boxes)
            print(scores)
            print(classes)
            print(valid_detections)
            
            # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
            #     cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)
            quit(0)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


