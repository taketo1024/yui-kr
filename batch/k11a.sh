#/bin/bash
#PBS -q C12-LONG
#PBS -l select=1:ncpus=48:mpiprocs=1
#PBS -l walltime=72:00:00
#PBS -M taketo.sano@riken.jp
#PBS -m abe
#PBS -j oe
#PBS -o /home/taketo.sano/logs/
#PBS -N yui-kr_k11a

echo "running yui-kr..."

source /home/taketo.sano/.cargo/env
cd /home/taketo.sano/yui-kr/

cargo update
cargo test -r --features batch_test --test k11a -- --test-threads=1 --show-output
