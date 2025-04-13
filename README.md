# Dense-Geometry-Supervision-for-Underwater-Depth-Estimation
This is an example of Adabins using [the module proposed in the paper](models/DepthTextureFusion.py). To put it simply, it involves passing the image through the proposed module to [obtain an enhanced texture image](EhanceImageGenModule.py), which is then fused with the features obtained from the decoder.

## Environment
The runtime environment can refer to the environment of [Adabins](https://github.com/shariqfarooq123/AdaBins), including the TXT file of relevant parameters.

## Rendered Datasets and Pretrained Models
The [datasets](我用夸克网盘分享了「uw_dataset.zip」，点击链接即可保存。打开「夸克APP」，无需下载在线播放视频，畅享原画5倍速，支持电视投屏。
链接：https://pan.quark.cn/s/05b939219de3
提取码：HUQJ) rendered in the paper and the [pre-trained weights](我用夸克网盘分享了「AdaBins_UW.pth」，点击链接即可保存。打开「夸克APP」，无需下载在线播放视频，畅享原画5倍速，支持电视投屏。
链接：https://pan.quark.cn/s/4a2fd573314e
提取码：kMmU) for the Adabins examples.
