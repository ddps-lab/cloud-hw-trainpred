chmod +x run_nvidia.sh
chmod +x run_workload.sh
nohup.out & nohup ./run_nvidia.sh >nohup1.out & nohup ./run_workload.sh >nohup2.out & 
