#!/bin/bash

remote_host=$1
remote_node=$2
filename=$3

if [ "$USER" != "root" ]; then
	echo "You must be root"
	exit 1
fi

if [ -z "$remote_host" ] || [ -z "$remote_node" ]; then
	echo "Usage: $0 <remote host> <remote node id> <filename>"
	exit 1
fi

dmesg -c > /dev/null

scp lgpu2rgpu-dma $remote_host:lgpu2rgpu-dma.$$
/opt/DIS/sbin/dis_tool control-dma-polling 0 3 1000 100
ssh $remote_host "/opt/DIS/sbin/dis_tool control-dma-polling 0 3 1000 100"
sleep 2

segment_size=1
repeats=20
modes="dma-push dma-pull"
gpus="0 1"

cat /proc/cmdline > $filename

for gpu in $gpus; do
	ssh $remote_host "killall lgpu2rgpu-dma.$$"
	ssh -f $remote_host "./lgpu2rgpu-dma.$$ --size=$segment_size -i --gpu=$gpu" 

	for mode in $modes; do
		echo >> $filename
		echo "remote gpu: $gpu, local ram, mode $mode" >> $filename
		./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode >> $filename
		dmesg -c | head -n 50 >> $filename

		for gpu2 in $gpus; do
			echo >> $filename
			echo "remote gpu: $gpu, local gpu: $gpu2, mode $mode" >> $filename
			./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode --gpu=$gpu2 >> $filename
			dmesg -c | head -n 50 >> $filename
		done
	done
	
	echo >> $filename
	echo >> $filename
	echo >> $filename
done


ssh $remote_host "killall lgpu2rgpu-dma.$$"
ssh -f $remote_host "./lgpu2rgpu-dma.$$ --size=$segment_size -i" 

for mode in $modes; do
	echo >> $filename
	echo "remote ram, local ram, mode $mode" >> $filename
	./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode >> $filename
	dmesg -c | head -n 50 >> $filename

	for gpu2 in $gpus; do
		echo >> $filename
		echo "remote gpu: $gpu, local gpu: $gpu2, mode $mode" >> $filename
		./lgpu2rgpu-dma --remote-node=$remote_node -i -c $repeats --bench=$mode --gpu=$gpu2 >> $filename
		dmesg -c | head -n 50 >> $filename
	done
done

ssh $remote_host "killall lgpu2rgpu-dma.$$"
ssh $remote_host "rm -f lgpu2rgpu-dma.$$"
