"""
Populates our smart contract with bridges and captchas
"""
import argparse
import base64
import json
import random
from string import ascii_lowercase, ascii_uppercase, digits
from captcha.image import ImageCaptcha
from web3 import Web3, HTTPProvider
from client import Client, W3

BRIDGES = list()
CAPTCHA_FILE = "captcha.png"
LOCAL_ACCOUNT = "account.json"
with open(LOCAL_ACCOUNT) as file:
    ACCOUNT_INFO = json.load(file)
    ACCOUNT = Web3.toChecksumAddress(ACCOUNT_INFO['address'].lower())
    PASS_PHRASE = ACCOUNT_INFO['pass_phrase']
    W3.eth.defaultAccount = ACCOUNT


def unlock_account():
    W3.personal.unlockAccount(ACCOUNT, PASS_PHRASE, duration=60)

def get_captcha():
    """
    Generates a captcha based on a random string

    Returns:
        captcha_data (string): the image data of the captcha
        captcha_answer (string): the answer for the captcha
    """
    captcha_answer = ''.join(random.choice(ascii_lowercase + ascii_uppercase + digits) for _ in range(10))
    image = ImageCaptcha()
    data = image.generate(captcha_answer)
    image.write(captcha_answer, CAPTCHA_FILE)
    with open(CAPTCHA_FILE, "rb") as captcha:
        captcha_data = base64.b64encode(captcha.read())

    return captcha_data.decode('utf-8'), captcha_answer

def load_bridges():
    for _ in range(10):
        captcha_data, captcha_answer = get_captcha()
        BRIDGES.append({
            "ip_address": "192.168.1.1",
            "captcha": captcha_data,
            "captcha_answer": captcha_answer
        }) 

"""
def serialize_bridges():
    intermediate_array = list()
    for bridge in BRIDGES:
        intermediate_array.append(bridge['ip_address'])
        intermediate_array.append(bridge['captcha'])
        intermediate_array.append(bridge['captcha_answer'])

    print(intermediate_array)
    return bytearray([bytes(x, 'utf-8') for x in intermediate_array])
"""


def populate_bridges(contract_client):
    unlock_account()

    for bridge in BRIDGES:

        tx_hash = contract_client.contract.functions.add_bridge(
            bridge["ip_address"],
            bridge["captcha"],
            bridge["captcha_answer"]
        ).transact({
            'chainId': 1016,
            'gas': 1000000,
            'nonce': W3.eth.getTransactionCount(W3.eth.accounts[0]) + 1
        })

        tx_reciept = W3.eth.waitForTransactionReceipt(tx_hash)
        print(tx_reciept)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interacts with our smart contract', add_help=True)
    parser.add_argument('-a', dest='address', help='The address of the contract', required=True)
    args = parser.parse_args()

    contract_address = args.address
    contract_client = Client(contract_address)
    load_bridges()
    populate_bridges(contract_client)