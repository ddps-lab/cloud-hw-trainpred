#!/bin/bash
# Get arguments
INSTANCE_TYPE=$1

IMAGE_ID="ami-0ea8263cbf553488e"
AWS_KEY="ys"
SUBNET_ID="subnet-f791a893"
SG_ID="sg-0f2dfaeaef1c8f808"
TAG = "ys" + INSTANCE_TYPE

# Launch instance & get informations
echo 'launch instance'
LAUNCH_INFO=$(aws ec2 run-instances --image-id $IMAGE_ID --count 1 --instance-type $INSTANCE_TYPE \
--key-name $AWS_KEY --subnet-id $SUBNET_ID --security-group-ids $SG_ID --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]')   
sleep 60
echo 'get instance info'
INSTANCE_ID=$(echo $LAUNCH_INFO | jq -r '. | .Instances[0].InstanceId')
INSTANCE_DNS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID | jq -r '. | .Reservations[0].Instances[0].PublicDnsName')
echo $INSTANCE_DNS

# Instance setting
# sleep 60
AWS_KEY="ys.pem"
# echo 'git clone and setting instance'
ssh -o "StrictHostKeyChecking no" -i $AWS_KEY ubuntu@$INSTANCE_DNS 'git clone https://github.com/hyoonseo159357/Collect-DCGMI.git'
ssh -i $AWS_KEY -t ubuntu@$INSTANCE_DNS 'cd /home/ubuntu/Collect-DCGMI/&& sudo bash ./settings.sh'

# # Run Experiments
# sleep 60
#AWS_KEY="ys.pem"
# echo 'start experiment'
#ssh -i $AWS_KEY -t ubuntu@$INSTANCE_DNS 'cd /home/ubuntu/Hardware-Data-Collect/&& python3.7 workload.py'
#ssh -i $AWS_KEY -t ubuntu@$INSTANCE_DNS 

# # Get results
# sleep 10
# AWS_KEY="ys.pem"
# scp -i /Users/heoyunseo/desktop/aws_pem/ys.pem -r ubuntu@$INSTANCE_DNS:/home/ubuntu/Hardware-Data2/ .

# # Terminate instance
# sleep 10
# echo 'terminate instance'
# TERMINATE_INFO=$(aws ec2 terminate-instances --instance-ids $INSTANCE_ID)
# echo $TERMINATE_INFO
