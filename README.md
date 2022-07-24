## Environment
- GPU: GTX1080Ti
- Ubuntu 16.04.4
- CUDA 10.0
- python 3.6
- torch 1.2.0
- torchvision 0.4.0
- cupy 6.2.0
- scipy 1.3.1
- pillow 6.1.0
- numpy 1.17.0

### training data 

1. Download Vimeo90k training data from [vimeo triplet dataset](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip).

For more informaiton about Vimeo90k, please refer to [TOFlow](https://github.com/anchen1011/toflow).


### Two-frame interpolation
1. To interpolate a frame between arbitrary two frames you have, run interpolate_twoframe.py with following command.

    ```bash
    python interpolate_twoframe.py 
    ```
2. Then you will have the interpolated output frame.

## Results
- Qualitativa results for large motion
![image](./results/tackle.png)
- Qualitativa results for occlusion
![image](./results/e-bike.png)

### Additional Results Video
- orignal video
![video](./results/fps30.gif)

- Interpolation result of video
![video](./results/fps60.gif)

## Acknowledgements
This code is based on [Lee/AdaCoF](https://github.com/HyeongminLEE/AdaCoF-pytorch)

