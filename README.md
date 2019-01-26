# face_recognition

1. add training images into "TrainDataset" (create folder by person name which will help to lable the training data.)

2. Run below command to crate pickle file.
  -- python encode_faces.py --dataset TrainDataset --encodings encodings.pickle

3. Run below command to recognize the face. 
  -- python faces_recognize_by_image.py --encodings encodings.pickle --image TestData/$image_name
