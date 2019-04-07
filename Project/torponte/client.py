import argparse
import json
import socket
from web3 import Web3, HTTPProvider

PROVIDER = "http://127.0.0.1:8545"
ABI_DIRECTORY = "build/contracts/Ponte.json"
W3 = Web3(HTTPProvider(PROVIDER))


class Client(object):
    """
    Our client to interact with the smart contract
    
    Attributes:
        contract_address (string): the address of the main contract
        contract (Contract): a Contract object that we can use to access functions
    """
    
    def __init__(self, contract_address):
        self.contract_address = contract_address

        with open(ABI_DIRECTORY) as f:
            info_json = json.load(f)
            abi = info_json["abi"]
            self.contract = W3.eth.contract(address=self.contract_address, abi=abi)

    def retrieve_captcha(self):
        """
        Gets most recent command from contract
        
        Returns:
            bridge_id (int): the id of a bridge
            captcha (string): a captcha tied to a bridge address
        """
        return self.contract.functions.get_bridge_captcha().call()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interacts with our smart contract bridge distributor', add_help=True)
    parser.add_argument('-a', dest='address', help='The address of the contract', required=True)
    args = parser.parse_args()

    contract_address = args.address
    client = Client(contract_address)

    bridge_id, captcha = client.retrieve_captcha()
    print(bridge_id, captcha)