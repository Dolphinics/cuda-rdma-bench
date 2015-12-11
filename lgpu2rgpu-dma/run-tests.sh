#!/bin/bash

remote_host=$1
remote_node=$2
filename=$3

if [ -z "$remote_host" ] || [ -z "$remote_node" ]; then
	echo "Usage: $0 <remote host> <remote node id> <filename>"
	exit 1
fi

scp lgpu2rgpu-dma $remote_host:

segment_size=1
repeats=20
modes="dma-push dma-pull"
gpus="0 1"

cat /proc/cmdline > $filename

for gpu in $gpus; do
	echo "REMOTE GPU $gpu" >> $filename
	ssh $remote_host "killall lgpu2rgpu-dma"
	ssh -f $remote_host "./lgpu2rgpu-dma --size=$segment_size -i --gpu=$gpu" 

	for mode in $modes; do
		echo >> $filename
		echo "ram mode $mode" >> $filename
		./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode >> $filename

		for gpu2 in $gpus; do
			echo >> $filename
			echo "gpu $gpu2 mode $mode" >> $filename
			./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode --gpu=$gpu2 >> $filename
		done
	done
	
	echo >> $filename
	echo >> $filename
	echo >> $filename
done


echo "REMOTE RAM" >> $filename
ssh $remote_host "killall lgpu2rgpu-dma"
ssh -f $remote_host "./lgpu2rgpu-dma --size=$segment_size -i" 

for mode in $modes; do
	echo >> $filename
	echo "ram mode $mode" >> $filename
	./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode >> $filename

	for gpu2 in $gpus; do
		echo >> $filename
		echo "gpu $gpu2 mode $mode" >> $filename
		./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode --gpu=$gpu2 >> $filename
	done
done

ssh $remote_host "killall lgpu2rgpu-dma"
