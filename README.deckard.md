# How to do training
 * checkout the code with my modifications
 * create a python2.7 conda venv per environment.py2.7_tf1.14.yml
 * copy our own training dataset at s3://deckard-data-science-assets-us-east-2/curated_datasets/street_number_recognition
   * train.tfrecord and val.tfrecord
   * see s3://deckard-data-science-assets-us-east-2/curated_datasets/street_number_recognitionreadme.txt for how these
   tfrecord files are prepared
 * if you want to try different trunk DNN, update model_type/feat_layers/strides values in config.py
 * source set_env.sh
    * basically add the pylib/src to PYTHONPATH
 * ./scripts/train.sh 0,1 24
     - 0,1 means use GPU0 and 1
     - 24 means for each batch, each GPU handle 24 images
 * the artifacts of the run will be created at ./checkpoint
 * open another terminal and run
    ```
        python scripts/eval_fscore.py
    ```  
    This script will periodically evaluate the recall/precision/f1_score of the
     latest checkpoint in ./checkpoint and save the result as tf_summary, which can be viewed with 
     tensorboard with other metrics saved by the main training process 
     
    Notice it will also save a fscore.csv in ./checkpoint.
 * If you want to use tensorboard to visualize the metrics,
    ```tensorboard --logdir checkpoint```

# How to test the trained model on new images  
 * If you need to get the detected boxes as data
  - first, run
  ```./scripts/test.sh ${GPU_ID} ${checkpoint_folder}/model.ckpt-xxx ${image_dir}```
   The script will output the detected boxes as IC15 label file (one for each image) in 
    ${checkpoint_folder}/test/model.ckpt-xxx/txt/*.txt
  
  - then optionally, you can use street_num_spotting/src/street_num_spotting/convert_pixellink_test_result_to_our_json_format.py
  to convert the IC15 label files to a json file (in our own label format)
  
  - then you can use street_num_spotting/notebook/recognition_result_reviewer.ipynb to review
  the detection result. You need to update following config values accordingly
    - config.image_base_dir, set to image_dir
    - config.recognition_result_json_fpath, set to the json label file created in the step above  
 
 * If you just want to see the detected text boxes rendered on the original images
    ```./scripts/test_any.sh ${GPU_ID} ${checkpoint_folder}/model.ckpt-xxx ${image_dir} ${output_dir}```
  The script will output jpgs with bounding box rendered into output_dir
  
  
# About the metrics saved in tf.summary
 * pixel_link_loss = prediction_loss_on_clones + regularization_loss
 * clone0/xxx_loss, the prediction loss component on clone 0