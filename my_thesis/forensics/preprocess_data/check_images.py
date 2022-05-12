import sys, os, shutil, glob

val_dir = "E:/2_Deep_Learning/Dataset/facial_forgery/dfdc/image/val/*/*/*"

val_paths = glob.glob(val_dir)
val_paths = [p.replace("\\", "/") for p in val_paths]

val_img_file = "forensics/preprocess_data/auxiliary/val_image_dfdc.txt"

print("val_paths", len(val_paths))

with open(val_img_file, "w") as f:
    for path in val_paths:
        fname = os.path.basename(path)
        ftype = path.split('/')[-3]
        f.write(ftype+'/'+fname)
        f.write('\n')