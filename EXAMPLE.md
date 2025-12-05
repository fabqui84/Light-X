## text-conditioned video relighting & recam (gradual mode)
```
python inference.py \
    --video_path 'test/videos/1.mp4' \
    --stride -1 \
    --out_dir results \
    --camera 'traj' \
    --mode 'gradual' \
    --mask \
    --target_pose 0 -20 0.2 0 0 \
    --traj_txt 'test/trajs/traj3.txt' \
    --relit_txt 'test/relit/relit2.txt' \
    --relit_vd \
    --relit_cond_type 'ic' \
    --recam_vd
```

https://github.com/user-attachments/assets/7dee55dc-2db0-4f0a-a8aa-c1dde9da0b9e


## text-conditioned video relighting & recam (direct mode)
```
python inference.py \
    --video_path 'test/videos/1.mp4' \
    --stride -1 \
    --out_dir results \
    --camera 'target' \
    --mode 'direct' \
    --mask \
    --target_pose 0 -20 0.2 0 0 \
    --traj_txt 'test/trajs/traj1.txt' \
    --relit_txt 'test/relit/relit2.txt' \
    --relit_vd \
    --recam_vd \
    --relit_cond_type 'ic'
```


https://github.com/user-attachments/assets/803c0de3-ab36-42f3-b7cc-dd9805cadb2a


## text-conditioned video relighting & recam (bullet time mode)
```
python inference.py \
    --video_path 'test/videos/1.mp4' \
    --stride -1 \
    --out_dir results \
    --camera 'traj' \
    --mode 'bullet' \
    --mask \
    --target_pose 0 -20 0.2 0 0 \
    --traj_txt 'test/trajs/traj1.txt' \
    --relit_txt 'test/relit/relit2.txt' \
    --relit_vd \
    --recam_vd \
    --relit_cond_type 'ic'
```


https://github.com/user-attachments/assets/d49d9ddf-5dd3-490f-beac-ba44a4cc67dd


## text-conditioned video relighting & recam (dolly-zoom mode)
```
python inference.py \
    --video_path 'test/videos/1.mp4' \
    --stride -1 \
    --out_dir results \
    --camera 'target' \
    --mode 'dolly-zoom' \
    --mask \
    --target_pose 0 0 0.5 0 0 \
    --traj_txt 'test/trajs/traj1.txt' \
    --relit_txt 'test/relit/relit2.txt' \
    --relit_vd \
    --recam_vd \
    --relit_cond_type 'ic'
```


https://github.com/user-attachments/assets/eaa8b077-c296-4615-aff3-1298f11c44d7


## background image-conditioned video relighting
```
VIDEO_DIR="test/videos/bg/2"
python tools/sam/sam_fg.py --input_video "${VIDEO_DIR}/input.mp4"
python inference.py \
    --video_path "${VIDEO_DIR}/input.mp4" \
    --stride 1 \
    --out_dir results \
    --mask \
    --relit_vd \
    --relit_txt 'test/relit/bg/relit1.txt' \
    --relit_cond_type 'bg'
```

https://github.com/user-attachments/assets/ac00f4ae-1f38-468f-aa3e-0f446056f415



## reference image-conditioned video relighting
```
python inference.py \
    --video_path test/videos/ref/2.mp4 \
    --stride 1 \
    --out_dir results \
    --mask \
    --relit_vd \
    --relit_cond_type 'ref' \
    --relit_cond_img test/cond/ref_cond/1.png \
    --transformer_path tqliu/Light-X-Uni
```


https://github.com/user-attachments/assets/c6a19897-d572-4ce3-9324-74ef3c9179e1


## hdr map-conditioned video relighting
```
python inference.py \
    --video_path test/videos/hdr/2.mp4 \
    --stride 1 \
    --out_dir results \
    --mask \
    --relit_vd \
    --relit_cond_type 'hdr' \
    --relit_cond_img test/cond/hdr_cond/1.exr \
    --transformer_path tqliu/Light-X-Uni
```


https://github.com/user-attachments/assets/44300195-da93-4622-ac05-af9ebe5e292f

