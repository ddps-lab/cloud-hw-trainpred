cuda :  11.4 
tensorflow : 2.7.0
N.virginia / Deep Learning AMI GPU CUDA 11.4.1 (Ubuntu 18.04) 20211204 AMI)

1. 로컬에서 파일 다운로드받기
2. 파일안에 pem키 넣어주기
3. sh startCLI.sh g4dn.xlarge

ssh에 접속되면
4. cd Hardware-Data2
5. sudo bash ./run_all.sh (gpu+cpu둘다수집)
   sudo bash ./run_gpu.sh (gpu만수집)


