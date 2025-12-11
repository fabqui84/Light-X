## ___***Light-X: Generative 4D Video Rendering with Camera and Illumination Control***___

<div align="center">
    <a href='http://tqtqliu.github.io/' target='_blank'>Tianqi Liu</a><sup>1,2,3</sup>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a><sup>1</sup>&emsp;       
    <a href='https://inso-13.github.io/' target='_blank'>Zihao Huang</a><sup>1,2,3</sup>&emsp;   
    <a href='https://daniellli.github.io/' target='_blank'>Shaocong Xu</a><sup>2</sup>&emsp;
    <a href='https://sainingzhang.github.io/' target='_blank'>Saining Zhang</a><sup>2,4</sup><br>
    <a href='https://hugoycj.github.io/' target='_blank'>Chongjie Ye</a><sup>5</sup>&emsp;
    <a href='https://arlo0o.github.io/libohan.github.io/' target='_blank'>Bohan Li</a><sup>6,7</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=396o2BAAAAAJ&hl=en' target='_blank'>Zhiguo Cao</a><sup>3</sup>&emsp;
    <a href='https://weivision.github.io/' target='_blank'>Wei Li</a><sup>1</sup>&emsp;
    <a href='https://sites.google.com/view/fromandto' target='_blank'>Hao Zhao</a><sup>4,2,*</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>1,*</sup>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, NTU&emsp;
    <sup>2</sup>BAAI&emsp;
    <sup>3</sup>HUST&emsp;
    <sup>4</sup>AIR,THU&emsp;
    <sup>5</sup>FNii, CUHKSZ&emsp;
    <sup>6</sup>SJTU&emsp;
    <sup>7</sup>EIT (Ningbo)
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2512.05115" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2512.05115-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
  </a>
  <a href="https://lightx-ai.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red">
  </a>
  <a href="https://youtu.be/ui9Lg2H--0c">
    <img src="https://img.shields.io/badge/YouTube-Video-blue?logo=youtube&logoColor=blue">
  </a>
  <a href='https://huggingface.co/datasets/tqliu/Light-Syn/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
  <a href='https://huggingface.co/tqliu/Light-X/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-red'></a>
  <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=TQTQliu.Light-X" alt="Visitors">
  </a>
</p>



>**TL;DR**: <em>Light-X is a video generation framework that jointly controls camera trajectory and illumination from monocular videos.</em>

https://github.com/user-attachments/assets/99a95a74-5b42-41b1-a84f-cdc2ea24b3c9

## 🌟 Abstract
Recent advances in illumination control extend image-based methods to video, yet still facing a trade-off between lighting fidelity and temporal consistency. Moving beyond relighting, a key step toward generative modeling of real-world scenes is the joint control of camera trajectory and illumination, since visual dynamics are inherently shaped by both geometry and lighting. To this end, we present **Light-X**, a video generation framework that enables controllable rendering from monocular videos with both viewpoint and illumination control. **1)** We propose a disentangled design that decouples geometry and lighting signals: geometry and motion are captured via dynamic point clouds projected along user-defined camera trajectories, while illumination cues are provided by a relit frame consistently projected into the same geometry. These explicit, fine-grained cues enable effective disentanglement and guide high-quality illumination. **2)** To address the lack of paired multi-view and multi-illumination videos, we introduce **Light-Syn**, a degradation-based pipeline with inverse-mapping that synthesizes training pairs from in-the-wild monocular footage. This strategy yields a dataset covering static, dynamic, and AI-generated scenes, ensuring robust training. Extensive experiments show that Light-X outperforms baseline methods in joint camera-illumination control and surpasses prior video relighting methods under both text- and background-conditioned settings.


## 🛠️ Installation
#### Clone Light-X
  ```
  git clone https://github.com/TQTQliu/Light-X.git
  cd Light-X
  ```
#### Setup environments
  ```
 conda create -n lightx python=3.10
 conda activate lightx
 pip install -r requirements.txt
  ```

#### Download Pretrained Models

Pretrained models are hosted on Hugging Face and load automatically during inference.  
If your environment cannot access Hugging Face, you may download them manually:

- Text-based / background-image lighting:  [tqliu/Light-X](https://huggingface.co/tqliu/Light-X)

- HDR / reference-image lighting (also supports text/bg):  [tqliu/Light-X-Uni](https://huggingface.co/tqliu/Light-X-Uni)

After downloading, specify the local model directory using `--transformer_path` in `inference.py`.


## 🚀 Inference
Run inference using the following script:
```bash
bash run.sh
```
All required models will be downloaded automatically.

We also provide **[EXAMPLE.md](EXAMPLE.md)** with commonly used commands and their corresponding visual outputs.
Please refer to this file to better understand the purpose and effect of each argument.


The `run.sh` script executes `inference.py` with the following arguments:

```bash
python inference.py \
    --video_path [INPUT_VIDEO_PATH] \
    --stride [VIDEO_STRIDE] \
    --out_dir [OUTPUT_DIR] \
    --camera ['traj' | 'target'] \
    --mode ['gradual' | 'bullet' | 'direct' | 'dolly-zoom'] \
    --mask \
    --target_pose [THETA PHI RADIUS X Y] \
    --traj_txt [TRAJECTORY_TXT] \
    --relit_txt [RELIGHTING_TXT] \
    --relit_cond_type ['ic' | 'ref' | 'hdr' | 'bg'] \
    [--relit_vd] \
    [--relit_cond_img CONDITION_IMAGE] \
    [--recam_vd]
```
#### Key Arguments:
🎥 Camera

- `--camera`:
  Camera control mode:
  - `traj`: Move the camera along a trajectory
  - `target`: Render from a fixed target view

- `--mode`:
  Style of camera motion when rendering along a trajectory:
  - `gradual`: Smooth and continuous viewpoint transition; suitable for natural, cinematic motion
  - `bullet`: Fast forward-shifting / orbit-like motion with stronger parallax
  - `direct`: Minimal smoothing; quickly moves from start to end pose
  - `dolly-zoom`: Hitchcock-style effect where the camera moves while adjusting radius; the subject stays the same size while the background expands/compresses

- `--traj_txt`: Path to a trajectory text file (required when `--camera traj` is used)

- `--target_pose`: Target view `<theta phi r x y>` (required when `--camera target` is used)

- `--recam_vd`: Enable video re-camera mode

  
See [here](https://github.com/TrajectoryCrafter/TrajectoryCrafter/blob/main/docs/config_help.md) for more details on camera parameters.

💡 Relighting

- `--relit_txt`: Path to a relighting parameter text file
- `--relit_vd`: Enable video relighting
- `--relit_cond_type`:
  Choose the lighting condition source:
  - `ic`: IC-Light (text-based / background-based lighting)
  - `ref`: Reference image lighting
  - `hdr`: HDR environment map lighting
  - `bg`: Background image lighting
- `--relit_cond_img`: Path to the conditioning image (required for `ref` / `hdr` modes)



## 🔥 Training

#### 1. Prepare Training Data

Download the [dataset](https://huggingface.co/datasets/tqliu/Light-Syn/tree/main) .

#### 2. Generate Metadata

Generate the metadata JSON file describing the training samples:

```bash
python tools/gen_json.py -r <DATA_PATH>
```

Then Update the `DATASET_META_NAME` in `train.sh` to the path of the newly generated JSON file.

#### 3. Start Training

```bash
bash train.sh
```

#### 4. Convert Zero Checkpoint to fp32

Convert the DeepSpeed ZeRO sharded checkpoint to a single fp32 file for inference.

_Example (for step 16000):_

```bash
python tools/zero_to_fp32.py train_outputs/checkpoint-16000 train_outputs/checkpoint-16000-out --safe_serialization
```

> `train_outputs/checkpoint-16000-out` is the resulting fp32 checkpoint directory.

You can then pass this directory directly to the inference script:

```bash
python inference.py --transformer_path train_outputs/checkpoint-16000-out
```


## 📚 Citation
If you find our work useful for your research, please consider citing our paper:

```
@article{liu2025light,
  title={Light-X: Generative 4D Video Rendering with Camera and Illumination Control},
  author={Liu, Tianqi and Chen, Zhaoxi and Huang, Zihao and Xu, Shaocong and Zhang, Saining and Ye, Chongjie and Li, Bohan and Cao, Zhiguo and Li, Wei and Zhao, Hao and others},
  journal={arXiv preprint arXiv:2512.05115},
  year={2025}
}
```

## ♥️ Acknowledgement
This work is built on many amazing open-source projects shared by [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), [IC-Light](https://github.com/lllyasviel/IC-Light), and [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun). Thanks all the authors for their excellent contributions!

## 📧 Contact
If you have any questions, please feel free to contact Tianqi Liu <b>(tq_liu at hust.edu.cn)</b>.
