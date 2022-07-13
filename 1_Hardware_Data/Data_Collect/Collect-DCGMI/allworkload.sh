for var1 in 32 64 128 224 256
  do
    for var2 in 16 32 64 128 256
    do
      #for var3 in 'LeNet5' 'VGGSmall' 'VGG19' 'ResNetSmall' 'MNIST_CNN' 'CIFAR10_CNN' 'AlexNet' 'InceptionV3' 'InceptionResNetV2' 'Xception' 'EfficientNetB0' 'MobileNetV2' 'ResNet50'
      for var3 in 'LeNet5' 'VGGSmall' 'VGG19' 'VGG13' 'VGG16' 'VGG11' 'ResNetSmall' 'MNIST_CNN' 'CIFAR10_CNN' 'AlexNet' 'InceptionV3' 'InceptionResNetV2' 'Xception' 'EfficientNetB0' 'MobileNetV2' 'ResNet50' 'ResNet18' 'ResNet34'     
      do
        python3.7 workload.py --model $var3 --dataset $var1 --batch_size $var2
      done
    done  
  done
 
