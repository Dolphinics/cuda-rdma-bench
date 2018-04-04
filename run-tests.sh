#!/bin/bash

bin=rdma-bench
max_log_lines=50
filename=$1
remote_host=$2
remote_node=$3
modes="dma-push dma-pull scimemwrite scimemcpy-write scimemcpy-read write read"

if [ "$USER" != "root" ]; then
	>&2 echo "You must be root!"
	exit 1
fi

# Check arguments
if [ -z "$filename" ] || [ -z "$remote_host" ] || [ -z "$remote_node" ]; then
	>&2 echo "Usage: $0 <log file> <remote host> <remote node id>"
	exit 1
fi

if [ -f "$filename" ]; then
	>&2 echo "File '$filename' already exists"
	exit 2
elif [ ! -f "$bin" ]; then
	>&2 echo "Can't find binary '$bin'"
	exit 2
fi

# Clear system log
dmesg -c > /dev/null

# Copy binary to remote system and set correct DMA polling mode
ctrl_dma_poll="/opt/DIS/sbin/dis_tool control-dma-polling 0 3 1000 100"
scp ${bin} ${remote_host}:${bin}.$$
$ctrl_dma_poll
ssh $remote_host "$ctrl_dma_poll"
sleep 2

# Test configuration
segment_sizes="1 10"
repeats=20
gpus=("" 0 1)

# Write time to log file
date > $filename
echo >> $filename
chmod 666 $filename

# Write kernel config to log file
local_host=`hostname`
echo -n "$local_host  " >> $filename
cat /proc/cmdline >> $filename
echo -n "$remote_host  " >> $filename
ssh $remote_host "cat /proc/cmdline" >> $filename
echo >> $filename

function run_test {
	# Extract arguments
	local segment_size=$1
	local bench_type=$2
	local local_gpu=$3
	local remote_gpu=$4

	local_gpu_arg=""
	if [ -n "${local_gpu}" ]; then
		local_gpu_arg="--gpu=$local_gpu"
	fi

	remote_gpu_arg=""
	if [ -n "${remote_gpu}" ]; then
		remote_gpu_arg="--gpu=$remote_gpu"
	fi

	# Run remote server
	ssh -f $remote_host "./${bin}.$$ --size=${segment_size} -i ${remote_gpu_arg}"

	# Run local client
	echo >> $filename
	./${bin} --remote-node=${remote_node} -i -c ${repeats} --bench=${bench_type} ${local_gpu_arg} >> $filename

	# Copy client log
	echo >> $filename
	echo $local_host >> $filename
	dmesg -c | head -n $max_log_lines >> $filename

	# Copy server log
	echo >> $filename
	echo $remote_host >> $filename
	ssh $remote_host "dmesg -c | head -n $max_log_lines" >> $filename

	ssh $remote_host "killall ${bin}.$$" &> /dev/null

	echo >> $filename
	echo >> $filename
	echo >> $filename
}

function run_all {
	for mode in $modes; do
		for size in $segment_sizes; do
			run_test $size $mode $1 $2
		done
	done
}

# Run tests
ssh $remote_host "killall ${bin}.$$" &> /dev/null

let i=0
for local_gpu in "${gpus[@]}"; do
	for remote_gpu in "${gpus[@]}"; do
		echo "**** #$i (local: '$local_gpu', remote: '$remote_gpu')" >> $filename
		run_all $local_gpu $remote_gpu
		echo "**** end #$i" >> $filename
		let i=i+1
	done
done

# Clean up remote end
ssh $remote_host "killall ${bin}.$$" &> /dev/null
ssh $remote_host "rm -f ${bin}.$$"
