#!/bin/bash

geth --exec "personal.unlockAccount(eth.accounts[0], \"con162ess\")" attach http://127.0.0.1:8545
