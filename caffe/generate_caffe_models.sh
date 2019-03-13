#!/bin/bash
# Script to generate the caffe models distributed

echo "Generating the caffe models ..."

cat voc12_weakly/model/vgg128_noup/Split_caffe_model_a* > voc12_weakly/model/vgg128_noup/train_iter_100000.caffemodel
cat voc12_weakly/model/vgg128_noup/Split_caffe_solver_a* > voc12_weakly/model/vgg128_noup/train_iter_100000.solverstate

#Generating md5sum

echo "Generating md5sum of the models ..."
md5sum voc12_weakly/model/vgg128_noup/train_iter_100000.caffemodel
echo "Original md5sum: 3917eb168cf09f239d68e190824a6853"



md5sum voc12_weakly/model/vgg128_noup/train_iter_100000.solverstate
echo "Original md5sum: 3ef5376311f1258e39933b1671c7de9e"



