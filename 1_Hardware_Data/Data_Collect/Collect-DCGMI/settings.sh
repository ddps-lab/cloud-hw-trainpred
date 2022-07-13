pip3 install tensorflow==2.7.0

sudo apt update
sudo killall apt apt-get
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a
sudo apt update

# DSTAT 설치 + 권한설정
sudo apt install dstat
chmod +x dcgmi_field.sh

# DCGMI설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update && sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm

# train
sudo bash ./allworkload.sh
