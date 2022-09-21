# NeRF

A re-implementation of Neural Radiance Fields (NeRF).

Install requirments: `pip install requirements.txt`

Download data: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

Update the data paths in config files `configs/*.txt`

## Usage

```
$ python3 main.py --config $config_path --gpu $gpu_id
# if you want to re-train the model: --no_reload
# if you only want to inference images: --render_only]
# if you only want to inference test views rather visualization views: --render_test
```