import deeplabcut

deeplabcut.video_inference_superanimal(
    videos=[r"C:\Users\gl_pc\Documents\collected_frames_Rat1_20200914_reencoded.mp4"],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_mobilenet_v3_large_fpn",
    scale_list=[400, 500, 600, 700, 800],
    max_individuals=1,
    video_adapt=False,
)