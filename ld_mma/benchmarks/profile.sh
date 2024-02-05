# refer to https://github.com/reed-lau/cute-gemm/blob/main/profile.sh
ncu=`which ncu`
sudo ${ncu} --csv --log-file a.csv --cache-control=all --clock-control=base --metrics gpu__time_duration.sum ./gemm-multi-stage

python3 stat-csv.py