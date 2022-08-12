#!/bin/bash

while true
do
        R1=`cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data`
        T1=`cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data`
        sleep 0.1
        R2=`cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data`
        T2=`cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data`
        TBPS=`expr $T2 - $T1`
        RBPS=`expr $R2 - $R1`
        TKBPS=`expr $TBPS \* 4 \* 8 / 1000 / 1000`
        RKBPS=`expr $RBPS \* 4 \* 8 / 1000 / 1000`
        echo "tx: $TKBPS Mo rx: $RKBPS Mo"
done
