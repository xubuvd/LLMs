source /etc/profile
pdsh -w ssh:10.208.111.45,10.208.112.209,10.208.109.54,10.208.110.235 "bash /mnt/lptest/xubu/postrain/stopall.sh"
pdsh -w ssh:10.208.111.45,10.208.112.209,10.208.109.54,10.208.110.235 "rm -rf /tmp/*"

