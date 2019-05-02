#!/bin/bash

geth --datadir=./ethdata \
     --networkid 101610 \
     --nodiscover \
     --targetgaslimit '9000000000000' \
     --rpc \
     --rpcport 8545 \
     --rpcaddr 127.0.0.1 \
     --rpccorsdomain "*" \
     --rpcapi "eth,net,web3,personal,miner" \
     console
