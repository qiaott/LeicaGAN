python main.py --cfg cfg/Bird/Main/EarlyGLAM/eval.yml
python Evaluation_Metrics/IS_Pytorch/Flower/inception_score.py
python rp_pre_extraction.py --cfg cfg/RP/1.Extract_text_feature/Bird/TI.yml
python rp_main.py --cfg cfg/RP/2.Caculate_RP/Flower/EarlyGLAM/eval_TI_Seg.yml