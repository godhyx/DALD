<div align="center">
<h1>DALD</h1>

****
</div>

![Logo](logo.png)
Our code is modified based on CLRerNet, so many folder names remain unchanged; we will address these issues in future updates.

## Prepare Dataset
Download [CULane](https://xingangpan.github.io/projects/CULane.html). To separate lane annotations into *near* (y ≤ 350) and *far* (y ≥ 350) categories, run the following script:
```
python split_lanes_by_distance_culane.py /path/to/CULane /path/to/output_dir
```


## Set Environment
Please refer to the original [CLRerNet](https://github.com/hirotomusiker/CLRerNet) repository for environment configuration instructions.

## Training
Train DALD with 1 GPU:
```
python tools/train.py configs/dald/culane/culane_dla34.py
```

## Evaluation
```
python tools/test.py configs/dald/culane/culane_dla34.py culane_dla34.pth
```



## Acknowledgements

[CLRerNet](https://github.com/hirotomusiker/CLRerNet)
