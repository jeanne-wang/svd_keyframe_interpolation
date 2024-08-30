# Keyframe Interpolation with Stable Video Diffusion
**[Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation
]()** 
</br>
[Xiaojuan Wang](https://jeanne-wang.github.io/),
[Boyang Zhou](https://www.linkedin.com/in/zby2003/),
[Brian Curless](https://homes.cs.washington.edu/~curless/),
[Ira Kemelmacher](https://www.irakemelmacher.com/),
[Aleksander Holynski](https://holynski.org/),
[Steve Seitz](https://www.smseitz.com/)
</br>
[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://svd-keyframe-interpolation.github.io/)

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input frame 1</td>
        <td>Input frame 2</td>
        <td>Generated video</td>
    </tr>
  	<tr>
	  <td>
	    <img src=examples/example_001/frame1.png width="250">
	  </td>
	  <td>
	    <img src=examples/example_001/frame2.png width="250">
	  </td>
	  <td>
	    <img src=examples/example_001.gif width="250">
	  </td>
  	</tr>
  	<tr>
	  <td>
	    <img src=examples/example_003/frame1.png width="250">
	  </td>
	  <td>
	    <img src=examples/example_003/frame2.png width="250">
	  </td>
	  <td>
	    <img src=examples/example_003.gif width="250">
	  </td>
  	</tr>
</table >


## Quick Start
### 1. Setup repository and environment
```
git clone https://github.com/jeanne-wang/svd_keyframe_interpolation.git
cd svd_keyframe_interpolation

conda env create -f environment.yml
```
### 2. Download checkpoint
Download the finetuned [checkpoint](https://drive.google.com/drive/folders/1H7vgiNVbxSeeleyJOqhoyRbJ97kGWGOK?usp=sharing), and put it under `checkpoints/`.
```
mkdir -p checkpoints/svd_reverse_motion_with_attnflip
cd checkpoints/svd_reverse_motion_with_attnflip
pip install gdown
gdown 1H7vgiNVbxSeeleyJOqhoyRbJ97kGWGOK --folder
```



### 3. Launch the inference script!
The example input keyframe pairs are in `examples/` folder, and 
the corresponding interpolated videos (1024x576, 25 frames) are placed in `results/` folder.
</br>
To interpolate, run:
```
bash keyframe_interpolation.sh
```
## Light-weight finetuing
```
The synthetic training videos dataset will be released soon.
```

