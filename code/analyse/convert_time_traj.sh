#!/bin/bash
for num in `seq 0 23`
do
echo handle data $num
python convert_id2loc.py --base_dir /home/xjm/MoveSim/data1/traffic_hash1/time --data_name $num.txt --out_name $num.json --dic_path /home/xjm/MoveSim/data1/traffic_hash1/id2loc.json --limit 100
done
echo "finish"