# MXNet_contrib
My modified version of detection-related operators for MXNet.

## MultiPrior

Usage: generate anchors(or prior boxes) for SSD detector.

The original MXNet implementation of Priorbox layer in SSD by [zhreshold](https://github.com/zhreshold/mxnet-ssd) only support single min_size param, which generates (#size + # ratio - 1) anchors per location. My modified version of MultiPrior Operator generate (#size x #ratio) anchors, which is more flexible.

## MultiProposalTarget

Usage: match proposed ROI with GT boxes and calculated its regression targets.

The [SNIPER](https://github.com/mahyarnajibi/SNIPER) method for multi-scale training use real pixel values to filter unvalid boxes in hyper-parameter `valid_range`, which is unapproiate for datasets whose image sizes vary in a large range(e.g. from 480p to 4k). I modified this `valid_range` into relative values, a.k.a. sqrt(box_area/image_area). This modification enable us to train a SOTA detector on industrial datasets.

## MultiboxDetection

Usage: apply regression results to prior boxes and perfrom NMS to yeild final detection results.

The original MXNet implementation of detection_out layer in SSD by [zhreshold](https://github.com/zhreshold/mxnet-ssd) rank scores across all classes, which cause inter-class competition of detection results. In this case, the class with less data are dominated by class with more data, which cause rather low mAPs for rare classes. My modified version sperate classes in sorting process and support different TOP_K hyper-parameter for each class. This modification protects rare classes from dominating by other classes and benefit overall mAP.

## AssignAnchor

Usage: perform matching between anchors and GT boxes, collect matching results.

This OP is built from scratch for collecting detailed matching stat for anchors. It return the matched GT class, IOU for each anchor, which provide an overview for matching results. I also developed a series of metrics for evaluating anchor hyper-parameters. With that, we can gain insights about these anchor settings before training.
