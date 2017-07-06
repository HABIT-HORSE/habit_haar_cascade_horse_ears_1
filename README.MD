
The HABIT haar cascade horse_ears 1 Haar Cascade detector

Developer: Dr Steve North - steve@stevenorth.com - 2017
 
This 'horse ears' detector was developed in 2017 for the HABIT (Horse Automated Behaviour Identification Tool) project [1]. 

The intention of building this detector was to investigate 'shortcuts' to training a detector for complex, deformable target objects with high levels of intra-class shape variability... such as the horse.

By identifying 'salient features' (distinctive, minimally changing physical attributes) of the target object, it might be possible to produce a simpler, faster, more easily trained detector. 
In this case, I was investigating the use of horses' ears as a salient feature for detecting horses in images and video.
 
The images used to train the Haar Cascade [2] may be found in the HABIT horse ears still image detector training dataset 904 [3]. This dataset includes a text file list of the negative images and a file containing the ROIs (Regions Of Interest) in the positive images.

Dataset 904 [3] has a total of 904 images, of which 200 are positive and 704 are negative.


Method to build and train the detector, using Python [4] and OpenCV [5].

1. The operating system used for development was: Microsoft Windows 7 Enterprise Service Pack 1 64-bit [Version 6.1.7601]. This process was also tested on Windows 8 and 8.1. Hardware: Intel Core i7 CPU / 3.06 Ghz / 6GB RAM
2. Pre requisites for the Windows / Python installation used: 
Python 2.7.11 (v2.7.11:6d1b6a68f775, Dec  5 2015, 20:32:19) [MSC v.1500 32 bit (Intel)] on win32
Open CV 2.4.11 (note: for OpenCV commands to work in Windows, the system environmental variable %path% needs to include the path to OpenCV’s bin directory)
Ffmpeg (comes with OpenCV, but may need configuring to work)
Installed python packages and versions:
Imutils 0.3.6
Matplotlib 1.2.0
Numpy 1.11.2
Pillow 3.4.2
Pip 8.1.1
Scikit-learn 0.18
Scipy 0.11.0
Setuptools 18.2
Six 1.10.0
Utility 1.0

3. Made a positives vector file (vector/facevector.vec):

The vector file contains resized thumbnails (20x20)of all 200 positive images in dataset 904 [3].
A .vec file is binary / hexadecimal in the format:
4 bytes denoting number of total images (int)
4 bytes denoting size of images (int)
2 bytes denoting min value (short)
2 bytes denoting max value (short)
example in hexadecimal: 6400 0000 4605 0000 0000 0000
	   	           #images size of h*w min     max
decimal	    	        100  1350	   0 		0

Ran this command from Command Prompt (or in a Windows batch file):
opencv_createsamples.exe -info positive/info.txt -vec vector/facevector.vec -num 200 -w 20 -h 20

Where:
positive/info.txt is a list of ROIs for the positive images (this comes with dataset 904 [3])
-num = the number of positive samples to generate
-w = the width (in pixels) of the output samples
-h  = the height (in pixels) of the output samples

Note: for OpenCV command to be recognised in Windows, the system environmental variable %path% needs to include the path to OpenCV’s bin directory.

4. Training the cascade:

The Haar Cascade object detection framework works by training a classifier, to find the most reliable combination of simple black/white graphical elemements (a Haar-like feature) for correctly detecting target objects. A window of a specified size is moved over the input image. For each subsection of the image, a corresponding feature is sought (this is achieved by calculating the number of pixels under both the black and white regions of the feature). This difference between black and white window regions is then compared to a learned threshold that separates non-objects from objects. Because each feature only has a detection ability that is slightly better than random chance, a large number of features are necessary to describe a target object with sufficient accuracy. Features are combined into training stages, with the detector becoming increasingly better at recognising the target object.

The first stage in preparing to train, is to run the following command from Windows Command Prompt (or in a Windows batch file):
opencv_traincascade.exe -data cascades -vec vector/facevector.vec -bg bg.txt -numStages 11 -minHitRate 0.999 -maxFalseAlarmRate 0.3 -numPos 185 -numNeg 1000 -w 20 -h 20 -precalcValBuffSize 1024 -precalcdxbufSize 1024
pause

Where: 

The default Haar feature-type is the default for opencv_traincascade.exe, with GAB (Gentle AdaBoost) .

Please note that GAB is a variant of the AdaBoost 'Adaptive Boosting' algorithm for machine learning proposed by Yoav Freund and Robert Schapire [6]. A 'weak learner' (or 'decision stump') is a classifier which performs relatively poorly. For example, its accuracy might be above chance, but only barely. In a Haar Cascade, each of the stages is an ensemble of 'weak learners', defined by graphical features. AdaBoost is adaptive in the sense that subsequent weak learners focus on those instances misclassified by previous classifiers. For example, only negative image samples that were classified mistakenly as positive by the last stage are passed to the next stage.

-data = the path for the output XML file
-vec = the vector file containing the positive image thumbnails
-bg = a text file listing the negative images
-numStages = the number of cascade training stages
-minHitRate = the minimal desired hit rate for each stage of the classifier. Overall hit rate may be estimated as (min_hit_rate ^ number_of_stages)
-maxFalseAlarmRate = maximum desired false alarm rate for each stage of the classifier. Overall false alarm rate may be estimated as (max_false_alarm_rate ^ number_of_stages)
-numPos = the number of positive sample images to use at each stage (this was set below the actual number of positive images in the vec file: 200, in order to allow for errors that might halt training). This is because opencv_traincascade.exe slowly increases the number of 'consumed' images as it goes through the training stages, in order to meet the -minHitRate and -maxFalseAlarmRate. When there are not enough images to consume, it crashes.
-numNeg = the number of negative images (this was set to 1000, although the actual number of negative images is 704. This is because numNeg is the the number of samples taken from the negative images, used in training at a given stage. They are picked randomly, cropped and scaled from the negative images. numNeg may be set higher - or lower - than the actual number of negative images, because the same image may get picked more than once. The samples that are picked are only those that were classified mistakenly as positive by the last stage).
-w = width of positive images
-h = height of positive images
-precalcValBuffSize = size of a memory buffer for precalculated feature values (in Mb).
-precalcdxbufSize = size of a memory buffer for precalculated feature indices (in Mb). The more memory you have the faster the training process.

Note: 
There is an older cascade training command in OpenCV: opencv_haartraining.exe
The more recent version is: opencv_traincascade.exe

opencv_traincascade and opencv_haartraining store the trained classifier / detector in different XML file formats. 

The newer OpenCV cascade detection interface (see CascadeClassifier class in objdetect module) supports both file formats. 

opencv_traincascade can save (export) a trained cascade in the older format. 
But opencv_traincascade and opencv_haartraining can not load (import) a classifier in another format for the further training after interruption.

Older format training (opencv_haartraining.exe) does not output the XML file directly. The output stage folders and text files have to be compiled into an XML file, using the OpenCV command cascade2XML. 

The new format training (opencv_traincascade.exe) creates an XML file, without this intermediate step.
 
Old opencv_haartraining.exe XML format header example:
<opencv_storage>
<haarcascade_frontalface_alt>
 
New opencv_traincascade.exe XML format header example:
<opencv_storage>
<cascade>
 
This horse ears detector is in the new XML format and this works with my other OpenCV code (that utilises the XML detector).

When I attempted to build this detector with the old opencv_haartraining.exe command (converting the output stage folders using cascade2XML), it generated a header
that didn't work with my other OpenCV code:
<opencv_storage>
<horse type_id="opencv-haar-classifier">
  <size>
    24 24</size>
  <stages>



In the directory specified by -data, a file called params.xml is created. The represents the training parameters, as specified when running the opencv_traincascade.exe command.
  
The file for training this detector looks like this:

<?xml version="1.0"?>
<opencv_storage>
<params>
  <stageType>BOOST</stageType>
  <featureType>HAAR</featureType>
  <height>20</height>
  <width>20</width>
  <stageParams>
    <boostType>GAB</boostType>
    <minHitRate>9.9900001287460327e-001</minHitRate>
    <maxFalseAlarm>3.0000001192092896e-001</maxFalseAlarm>
    <weightTrimRate>9.4999999999999996e-001</weightTrimRate>
    <maxDepth>1</maxDepth>
    <maxWeakCount>100</maxWeakCount></stageParams>
  <featureParams>
    <maxCatCount>0</maxCatCount>
    <featSize>1</featSize>
    <mode>BASIC</mode></featureParams></params>
</opencv_storage>


Training works until the error level is low enough, as measured by the falseAlarm rate.

Outcome when generating the 'horse ears' XML detector / classifier:

Data provided during each training stage using opencv_traincascade.exe -

In the columns:
N: Number of Haar features being tried
HR: Hit Rate based on the stage threshold
FA: False Alarm based on the stage threshold

At the end of each stage:
POS count: the number of positive image samples 'consumed' during this stage, compared to the total number of available positive samples.
NEG count acceptanceRatio: the number of negatives classified as positive divided by those which classified correctly as negatives in each stage. For example 1/1000 means that randomly picking 1000 windows of negatives from the negative images one of them is classified as positive.

The Haar Cascade (using AdaBoost) has a predermined number of stages (as specified in opencv_traincascade.exe's -numStages parameter).

At each stage, numPos and numNeg samples are used. It is important to understand that the numPos samples are the number of positive samples that are used as training samples in the current stage. This is different from the total number of samples in the vec file (vector/facevector.vec). For example, you might choose to use 0.95 * (number of samples in the vec file).

Similarly, the numNeg samples are the number of samples used in training of the current stage. They are picked randomly (cropped and scaled from the negative images) and they could be more or less than the total number of negative images that you have. For example, if there are 1000 negative images, numNeg could still be set to 5000 samples.

The negative samples that are picked are only those that were mistakenly classified as positive by the current-1 stage. This ensures that only the more difficult negatives go on to the next stage.

########################################################################

CONCLUSION TO TRAINING:

Training stages completed: 11 stages (State 0 - Stage 10)
Time to complete up to Stage 8: 20 minutes and 25 seconds.
Time taken for detector / classified training to complete: 1 hour 21 minutes and 7 seconds (this is fast!!) 

At the end of Stage 10, it was using 14 Haar features, with a HR of 1 and a FA rate of 0.206.
POS count: consumed 185 of 185
Neg count: acceptanceRatio 1000 : 8.93301e-006

The final XML detector / classifier file (cascade.xml) was found in the directory: /cascades/.

See example images of the detector being applied in the directory:  example test images showing detection

#########################################################################

TO DO FOR FUTURE VERSIONS:

1. Consider adding more positive images with variable background. The detector performs well on light backgrounds, but is less efficient against darker backgrounds.

[1] Steve North, Carol Hall, Amanda Roshier, and Clara Mancini, 2015. HABIT: Horse Automated Behaviour Identification Tool – A Position Paper. In Proceedings of the proceedings of ACI@BHCI (Animal Computer Interaction Workshop), British HCI 2015 (Lincoln, UK, 13 July 2015), BCS, UK.  http://dx.doi.org/10.13140/RG.2.1.3395.0881.

[2] P. Viola and M. Jones, 2001. Rapid object detection using a boosted cascade of simple features. In Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 511-518.  http://dx.doi.org/10.1109/CVPR.2001.990517.

[3] Steve North, 2017. HABIT horse ears still image detector training dataset 904.  http://dx.doi.org/10.5281/zenodo.823168.

[4] Python Software Foundation. 2017. Software Program: Python - programming language. Available from: https://www.python.org/.

[5] OpenCV team. 2017. Software Program: OpenCV (Open Source Computer Vision Library). Available from: http://opencv.org.

[6] Yoav Freund and Robert Schapire, 1999. A short introduction to boosting. Journal-Japanese Society For Artificial Intelligence 14, 771-780.
