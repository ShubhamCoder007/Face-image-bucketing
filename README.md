# Face-image-bucketing
Register face of a person using webcam or images. It helps you identify the person and bucket to respective directory identified as that registered individual.

Workflow overview
====================
We first detect the faces in images using MTCNN/RetinaFace which gives us an effective bounding box which has
the face as the region of interest.

Then we embed the image with the person's name to use it later for the match.
We use FaceNet/ArcFace for embedding.
For registering the person we can leverage both from image as well as webcam.
Webcam takes burst of 20 images and embeds them.
From image takes list of variable images, then derives mean out of this, and for rest 19 embedding vectors
it takes the most distant similarity from the mean vector in order to capture the outlier cases for the image.

We retrieve by computing the max of the similarities of all the embedding for every person.

After registering a person, threshold tuning is performed by equal error rate (false acceptance rate = false rejection rate)
post this threshold is updated in our config file.

finally for sorting image from a pool of input images we can simply direct the input directory and the output directory,
folder by the name of registered person name which is detected is created and images are streamed over there.


Usage
===================

Case 1. Register a person from webcam
we simply use 
```python capture_face.py```

Case 2. Register a person from images
```python register_from_images.py -n "your name" -i "input_location"```

Case 3. Run a realtime recognition among the registered faces
```python realtime_recognition.py```

Case 4. Sort images from pool of image to output directory having individual registered people
```python image_bucket_sorter.py -i "input_dir" -o "output_dir"```