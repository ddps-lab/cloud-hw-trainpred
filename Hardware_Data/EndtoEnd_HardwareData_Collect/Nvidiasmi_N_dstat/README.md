cuda11.4
tensorflow 2.7.0

N.virginia / Deep Learning AMI GPU CUDA 11.4.1 (Ubuntu 18.04) 20211204 AMI)

- 로컬에서 파일 다운로드받고, 파일안에 pem 키넣고
- sh startCLI.sh g4dn.xlarge

다깔고 ssh 접속되면
- cd Hardware-Data2
- sudo bash ./run_all.sh (gpu+cpu둘다수집)
- sudo bash ./run_gpu.sh (gpu만수집)


