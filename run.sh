# text-conditioned video relighting & recam (gradual mode)
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

# text-conditioned video relighting & recam (direct mode)
# python inference.py \
#     --video_path 'test/videos/1.mp4' \
#     --stride -1 \
#     --out_dir results \
#     --camera 'target' \
#     --mode 'direct' \
#     --mask \
#     --target_pose 0 -20 0.2 0 0 \
#     --traj_txt 'test/trajs/traj1.txt' \
#     --relit_txt 'test/relit/relit2.txt' \
#     --relit_vd \
#     --recam_vd \
#     --relit_cond_type 'ic'

# text-conditioned video relighting & recam (bullet time mode)
# python inference.py \
#     --video_path 'test/videos/1.mp4' \
#     --stride -1 \
#     --out_dir results \
#     --camera 'traj' \
#     --mode 'bullet' \
#     --mask \
#     --target_pose 0 -20 0.2 0 0 \
#     --traj_txt 'test/trajs/traj1.txt' \
#     --relit_txt 'test/relit/relit2.txt' \
#     --relit_vd \
#     --recam_vd \
#     --relit_cond_type 'ic'

# text-conditioned video relighting & recam (dolly-zoom mode)
# python inference.py \
#     --video_path 'test/videos/1.mp4' \
#     --stride -1 \
#     --out_dir results \
#     --camera 'target' \
#     --mode 'dolly-zoom' \
#     --mask \
#     --target_pose 0 0 0.5 0 0 \
#     --traj_txt 'test/trajs/traj1.txt' \
#     --relit_txt 'test/relit/relit2.txt' \
#     --relit_vd \
#     --recam_vd \
#     --relit_cond_type 'ic'

# background image-conditioned video relighting
# VIDEO_DIR="test/videos/bg/2"
# python tools/sam/sam_fg.py --input_video "${VIDEO_DIR}/input.mp4"
# python inference.py \
#     --video_path "${VIDEO_DIR}/input.mp4" \
#     --stride 1 \
#     --out_dir results \
#     --mask \
#     --relit_vd \
#     --relit_txt 'test/relit/bg/relit1.txt' \
#     --relit_cond_type 'bg'

# reference image-conditioned video relighting
# python inference.py \
#     --video_path test/videos/ref/2.mp4 \
#     --stride 1 \
#     --out_dir results \
#     --mask \
#     --relit_vd \
#     --relit_cond_type 'ref' \
#     --relit_cond_img test/cond/ref_cond/1.png \
#     --transformer_path tqliu/Light-X-Uni

# hdr map-conditioned video relighting
# python inference.py \
#     --video_path test/videos/hdr/2.mp4 \
#     --stride 1 \
#     --out_dir results \
#     --mask \
#     --relit_vd \
#     --relit_cond_type 'hdr' \
#     --relit_cond_img test/cond/hdr_cond/1.exr \
#     --transformer_path tqliu/Light-X-Uni