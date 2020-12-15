# Speed Detection Repository

Peronal project inspired by the Comma AI speed challenge:
github.com/commaai/speedchallenge

-----
Your goal is to predict the speed of a car from a video.
- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
-----

Installation instructions:
- If necessary, `pip3 install --upgrade pip`
- clone this repo
- download data from the linked repo
- `make create_environment`
- `conda activate speed_detection`
- `make requirements` (adjust Makefile if not Cuda 10.2)

-----

Please run nbs/data_processing to set up data, which also runs a segmentation network over all data to be used for image masking.
Finalized notebooks: fixed_point_regression, SuperPoint_fixed_point_regression.

Our approach ultimately uses SuperPoint for feature point tracking, and normalizes the found vectors that appear only on the road in front of the car.
A regression model is fit to the median of these tracks for each image pair. No classical image processing algorithms are needed such as optical flow or corner detectors.



