#/bin/bash
#PBS -q F
#PBS -l select=1:ncpus=64:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -M taketo.sano@riken.jp
#PBS -m abe
#PBS -j oe
#PBS -o /home/taketo.sano/logs/
#PBS -N yui-kr_k11a_354

echo "running yui-kr..."

source /home/taketo.sano/.cargo/env
cd /home/taketo.sano/yui-kr/

cargo update
cargo run -r -- 11a_354 -i bigint -s -p --debug
