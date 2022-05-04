chmod +x run_dstat.sh
chmod +x run_nvidia.sh
chmod +x run_workload.sh
nohup ./run_dstat.sh >nohup.out & nohup ./run_nvidia.sh >nohup2.out & nohup ./run_workload.sh >nohup3.out & 
