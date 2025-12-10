
# CUDA_VISIBLE_DEVICES=2 python train.py arg_.py
# CUDA_VISIBLE_DEVICES=1,2 nohup python train.py --num-gpus 2  --config-file ./myconfig/crack/maskformer2_R50_bs16_90k.yaml > output_crack.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0,2 nohup python train.py --num-gpus 2  --config-file ./myconfig/crack/maskformer2_R50_bs16_90k.yaml > output_crack.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --num-gpus 1   --config-file ./myconfig/crack/maskformer2_R50_bs16_90k.yaml

# python train.py \
#   --config-file ./myconfig/crack/maskformer2_R50_bs16_90k.yaml \
#   --eval-only MODEL.WEIGHTS ./output/model_0000499.pth
#     "--config-file", default="", metavar="FILE"
#     "--resume",  action="store_true"
#     "--eval-only", action="store_true"
#     "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
#     "--num-machines", type=int, default=1, help="total number of machines"
#     "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
#     "--dist-url",  default="tcp://127.0.0.1:{}".format(port),  help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.",
#     "opts",  help="""Modify config options at the end of the command. For Yacs configs, use space-separated "PATH.KEY VALUE" pairs. 
#                       For python-based LazyConfig, use "path.key=value".""",  default=None, nargs=argparse.REMAINDER,


# CUDA_VISIBLE_DEVICES=2 nohup python train.py --num-gpus 1  --config-file ./myconfig/elec/instance/maskformer2_R50_bs16_90k.yaml > output_componet.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2  python train.py --num-gpus 1  --config-file ./myconfig/elec/instance/maskformer2_R50_bs16_50ep.yaml 
# python train.py --num-gpus 4  --config-file ./myconfig/elec/instance/maskformer2_R50_bs16_50ep.yaml 
python train.py \
  --config-file ./myconfig/elec/instance/maskformer2_R50_bs16_50ep.yaml    \
  --eval-only MODEL.WEIGHTS ./output_component/model_0019999.pth

python demo.py \
  --config-file ../myconfig/elec/instance/maskformer2_R50_bs16_50ep.yaml    \
  --output ./elec_out \
  --input /home/zyc/projects_dir/project_temp/course_image_segmentation/Mask2Former-main/demo/elec_input/64901629522_25_scene_000002_000686_v1.png