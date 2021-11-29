mkdir -p $2
scp -P 23341 -r root@172.20.0.66:/content/data/$1/\*.jpg $2
