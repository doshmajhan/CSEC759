pragma solidity >=0.4.21 <0.6.0;

contract Ponte {
    address public owner;

    struct User {
        uint bridge_count;
        bytes public_key;
        bytes bridge_encrypted;
    }

    mapping(address => User) public users;

    constructor() public {
        owner = msg.sender;
    }

    function upload_key(bytes memory public_key) public {
        User storage user = users[msg.sender];
        user.public_key = public_key;
    }

    function give_bridge_to_user(address user_address, bytes memory bridge_encrypted) public {
        User storage user = users[user_address];
        user.bridge_encrypted = bridge_encrypted;
        user.bridge_count += 1;
    }
}
