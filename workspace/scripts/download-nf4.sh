modelscope download \
       --model DiffSynth-Studio/MiniMax-H3-NF4 \
       --local_dir /scratch/grzhu/weights/video/MiniMax-H3-NF4 \
       --include "minimax-h3-fl2va-nf4.safetensors,minimax-h3-ref2va-nf4.safetensors,minimax-h3-text-encoder-nf4.safetensors,video_vae_nf4.safetensors,audio_vae_nf4.safetensors"