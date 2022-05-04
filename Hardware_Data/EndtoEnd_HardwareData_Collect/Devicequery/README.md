# Save devicequery result as csv

- Original_deviceQuery : origianl deviceQuery.cpp (위치는 우분투기준 /usr/local/cuda-11.4/samples/1_Utilities/deviceQuery)
- Outputcsv_deviceQuery : deviceQuery.cpp file that saves the devicequery result as csv


< Make csv file and permission setting >
  
```
sudo touch devicequery.csv // devicequery 값을 저장해줄 csv 파일 생성
sudo chmod ugo+rwx devicequery.csv // read only -> write & read only
``` 


< Compile >

1. deviceQuery.cpp 수정
2. sudo make ( 에러나면 sudo make clean )
3. ./deviceQuery


< 추가해준 부분 간단히 >

```
#include <fstream> 
std::ofstream myfile;
myfile.open ("devicequery.csv");
myfile << "Maximum Texture Dimension Size (x,y,z)," << deviceProp.maxTexture1D;
myfile.close();
```

finally save the devicequery.csv locally
```
scp -i /Users/heoyunseo/desktop/aws_pem/ys.pem -r ubuntu@1.11.11.11:/usr/local/cuda-11.4/samples/1_Utilities/deviceQuery/devicequery.csv  .
```  
  
  
