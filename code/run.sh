#!/bin/bash
for video in in_paths = ../../dataset/big-bot/avi_rotated_clipped/*.avi; do
	# Checks if a file like instance
	# actually exists or is just a glob
	[ -e "$video" ] || continue
	# echo $out_name
	# Slow step so check if output file already exists
	out_name="../../results/subSpace/big-bot-out/$(basename "$video" .avi)_subSpace_out.avi"
	if [ -f "$out_name" ]; then
		echo -e $"Results for video $video exist, ... skipping run\n"
	else
	echo -e $"Currently running on $video\n"
  python3 subSpace.py -i "$video" -o "$out_name"
	fi
done

for video in in_paths = ../../dataset/SIGGRAPH-dataset/unstable/*.avi; do
	# Checks if a file like instance
	# actually exists or is just a glob
	[ -e "$video" ] || continue
	# echo $out_name
	# Slow step so check if output file already exists
	out_name="../../results/subSpace/SIGGRAPH-out/$(basename "$video" .avi)_subSpace_out.avi"
	if [ -f "$out_name" ]; then
		echo -e $"Results for video $video exist, ... skipping run\n"
	else
	echo -e $"Currently running on $video\n"
  python3 subSpace.py -i "$video" -o "$out_name"
	fi
done
