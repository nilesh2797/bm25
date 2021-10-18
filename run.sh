set -x
dataset=$1
res_dir=./Results/$dataset
data_dir=/mnt/c/nilesh/Datasets/$dataset
num_thread=48
K=100

mkdir -p $res_dir

/usr/bin/time ./bm25 	-X_Xf $data_dir/random100000_tst_X_Xf.txt \
						-Y_Yf $data_dir/Y_Yf.txt.bin \
						-Xf_Yf ../Results/$dataset/model/direct_Xf_Yf.bin \
						-out $res_dir/score_mat.bin \
						-num_thread $num_thread \
						-K $K \

# g++ -std=c++11 -O3 -pthread -fopenmp bm25.cpp -o bm25