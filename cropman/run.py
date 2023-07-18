import os

models = ['girl9','girl10']
for model in models:
    raw_root = f"/data/heyue/104/dataset/{model}_raw/frames"
    ori_root = raw_root + "_ori"
    # os.system(f"mv {raw_root} {ori_root}")
    # os.system(f"mkdir {raw_root}")
    for thing in os.listdir(ori_root):
        input_path = os.path.join(ori_root, thing)
        output_path = os.path.join(raw_root, thing)
        print(f"/data/heyue/makeup_related/cropman/app-console.py {input_path} 90 90 {output_path}")


# os.system(f"/data/heyue/makeup_related/cropman/app-console.py input.jpg 90 90 output.jpg")