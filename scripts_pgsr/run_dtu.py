import os

# scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scenes = [40]
data_base_path='G:\Dataset\DTU\DTU_from_neus'
# out_base_path='results/dtu_official'
out_base_path = 'G:/sources2024/fast-gs/output/3ea15a00-1'
eval_path='G:\Dataset\DTU\Official_DTU_Dataset'
out_name='test'
gpu_id=0

for scene in scenes:
    
    # cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    # print(cmd)
    # os.system(cmd)

    # cmd = f'python train.py -s {data_base_path}/dtu_scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} -r2 --ncc_scale 0.5'
    # print(cmd)
    # os.system(cmd)

    # cmd = f'python scripts/render_dtu.py -m {out_base_path}/dtu_scan{scene}/{out_name}'
    # print(cmd)
    # os.system(cmd)

    cmd = f'python scripts_pgsr/dtu_eval.py --data {out_base_path}/mesh/tsdf_fusion.ply --scan {scene} --mode mesh --dataset_dir {eval_path} --vis_out_dir {out_base_path}/mesh'
    print(cmd)
    os.system(cmd)