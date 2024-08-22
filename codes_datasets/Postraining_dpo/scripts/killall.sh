source /etc/profile
pdsh -w ssh:15.108.121.45,15.108.121.46,15.108.121.47,15.108.121.48 "bash /mnt/lptest/xubu/postrain/scripts/stopall.sh"

