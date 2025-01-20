# ADSUNet: An Accumulation-Difference-Based Siamese U-Shape Network for inter-frame Infrared Dim and Small Target Detection


Here, we provide the pytorch implementation of the paper: ADSUNet: An Accumulation-Difference-Based Siamese U-Shape Network for inter-frame Infrared Dim and Small Target Detection.


Please see `requirements.txt` for the requirements.


## Test

we provide the test data in "./testdata/Sequence1/new“

Test code:

```
test_demo.py
```


## Train on your data
Data structure

```
./IRSTD/inter_frame_data

"""
Data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the dataset.


## Main train code

```
main_ADSUnet.py
```
with the parameter: --gpu_ids=0 --net_G=SiamUnet_conc_diff_cbam --loss=miou --batch_size=32




