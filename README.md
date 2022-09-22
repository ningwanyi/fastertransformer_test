# fastertransformer_test
This repo is to test [FasterTransformer](https://github.com/NVIDIA/FasterTransformer.git) on PyTorch.

## Setup
- Clone FasterTransformer repo:
```
git clone https://github.com/NVIDIA/FasterTransformer.git
```
- Build docker image:
```
cd FasterTransformer
docker build -t faster_transformer:v1 -f docker/Dockerfile.torch .
```
- Run container:
```
docker run -itd --gpus all --name fastertrans -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -p 822:22 faster_transformer:v1
docker exec -it fastertrans /bin/bash 
```

## Running
- Build with PyTorch
```
cd /workspace/FasterTransformer/cmake
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON ..
make
```
- Install transformers
```
pip install transformers==2.5.1
```
- Generate the gemm_config.in file. Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)
```
cd /workspace/FasterTransformer/build

# ./bin/bert_gemm <batch_size> <sequence_length> <head_number> <size_per_head> <is_use_fp16> <int8_mode>
./bin/bert_gemm 1 32 12 64 1 0
```
- Run the PyTorch BERT sample:
```
# python ../examples/pytorch/bert/bert_example.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <--fp16> <--int8_mode 0/1/2/3> <--sparse> <--time>
python ../examples/pytorch/bert/bert_example.py 1 12 32 12 64 --time --int8_mode 1
```
