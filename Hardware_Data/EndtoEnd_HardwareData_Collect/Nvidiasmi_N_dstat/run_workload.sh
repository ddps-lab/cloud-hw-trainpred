for var1 in 32 64 128 224 256
  do
    for var2 in 16 32 64 128 256
    do
      for var3 in 'LeNet5' 'VGGSmall' 'VGG19' 'ResNetSmall' 'MNIST_CNN' 'CIFAR10_CNN' 'AlexNet' 'InceptionV3' 'InceptionResNetV2' 'Xception' 'EfficientNetB0' 'MobileNetV2' 'ResNet50'
      do
        python3.7 workload.py --model $var3 --dataset $var1 --batch_size $var2
      done
    done  
  done
  
# python3.7 workload.py --model 'LeNet5' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGGSmall' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGG11' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGG13' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGG16' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGG19' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNetSmall' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNet18' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNet34' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'AlexNet' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'InceptionV3' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'Xception' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'MobileNetV2' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNet50' --dataset 32 --batch_size 32

# python3.7 workload.py --model 'LeNet5' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'VGGSmall' --dataset 32 --batch_size 32       
# python3.7 workload.py --model 'VGG19' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNetSmall' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'AlexNet' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'InceptionV3' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'Xception' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'MobileNetV2' --dataset 32 --batch_size 32    
# python3.7 workload.py --model 'ResNet50' --dataset 32 --batch_size 32

# python3.7 workload.py --model 'LeNet5' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'VGGSmall' --dataset 32 --batch_size 128       
# python3.7 workload.py --model 'VGG19' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'ResNetSmall' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'AlexNet' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'InceptionV3' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'Xception' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'MobileNetV2' --dataset 32 --batch_size 128    
# python3.7 workload.py --model 'ResNet50' --dataset 32 --batch_size 128

# python3.7 workload.py --model 'LeNet5' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'VGGSmall' --dataset 64 --batch_size 64       
# python3.7 workload.py --model 'VGG19' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'ResNetSmall' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'AlexNet' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'InceptionV3' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'Xception' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'MobileNetV2' --dataset 64 --batch_size 64    
# python3.7 workload.py --model 'ResNet50' --dataset 64 --batch_size 64

# python3.7 workload.py --model 'LeNet5' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'VGGSmall' --dataset 224 --batch_size 32       
# python3.7 workload.py --model 'VGG19' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'ResNetSmall' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'AlexNet' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'InceptionV3' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'Xception' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'MobileNetV2' --dataset 224 --batch_size 32    
# python3.7 workload.py --model 'ResNet50' --dataset 224 --batch_size 32

# python3.7 workload.py --model 'LeNet5' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'VGGSmall' --dataset 128 --batch_size 128       
# python3.7 workload.py --model 'VGG19' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'ResNetSmall' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'MNIST_CNN' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'CIFAR10_CNN' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'FLOWER_CNN' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'AlexNet' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'InceptionV3' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'InceptionResNetV2' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'Xception' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'EfficientNetB0' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'MobileNetV2' --dataset 128 --batch_size 128    
# python3.7 workload.py --model 'ResNet50' --dataset 128 --batch_size 128

  
      
