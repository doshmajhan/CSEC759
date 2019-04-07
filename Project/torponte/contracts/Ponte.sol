pragma solidity >=0.4.21 <0.6.0;

import "./Seriality.sol";

contract Ponte is Seriality {

    struct Bridge {
        string ip_address;
        string captcha;
        string captcha_answer;
    }
    Bridge[] bridges;
    address public owner;
    uint nonce;

    constructor() public {
        owner = msg.sender;
    }

    function random() internal returns (uint) {
        uint random_number = uint(keccak256(abi.encodePacked(now, msg.sender, nonce))) % 10;
        nonce++;
        return random_number;
    }

    function compare_strings (string memory a, string memory b) internal pure returns (bool) {
       return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }

    /// @notice Updates the bridges mapping with new bridge addresses
    /// @param new_bridges The new bridges
    function update_bridges(bytes memory new_bridges) public {
        require(msg.sender == owner, "Must be owner");
        
        uint offset = 64*10;
        new_bridges = new  bytes(offset);

        for (uint i = 0; i < 10; i++) {
            string memory ip_address = new string(32);
            string memory captcha = new string(32);
            string memory captcha_answer = new string(32);

            bytesToString(offset, new_bridges, bytes(ip_address));
            offset -= sizeOfString(ip_address);

            bytesToString(offset, new_bridges, bytes(captcha));
            offset -= sizeOfString(captcha);

            bytesToString(offset, new_bridges, bytes(captcha_answer));
            offset -= sizeOfString(captcha_answer);

            ///Bridge memory bridge = Bridge(ip_address, captcha, captcha_answer);
            ///bridges.push(bridge);
        }
    }

    /// @notice Adds a single bridge to the list
    /// @param ip_address The ip address of the new bridge
    /// @param captcha The captcha data for the bridge
    /// @param captcha_answer The answer to the captcha
    function add_bridge(string memory ip_address, string memory captcha, string memory captcha_answer) public {
        require(msg.sender == owner, "Must be owner");
        Bridge memory bridge = Bridge(ip_address, captcha, captcha_answer);
        bridges.push(bridge);
    }

    /// @notice gets the captcha data and ID for a random bridge
    function get_bridge_captcha() public returns (uint, string memory) {
        uint random_index = random();
        Bridge memory bridge = bridges[random_index];
        return (random_index, bridge.captcha);
    }

    /// @notice takes in the answer for a bridge specified by ID and checks if its correct
    /// @param bridge_id The index of the bridge in the list
    /// @param captcha_answer the answer contained within the captcha
    function submit_bridge_captcha(uint bridge_id, string memory captcha_answer) public view returns (string memory) {
        Bridge memory bridge = bridges[bridge_id];
        if (compare_strings(bridge.captcha_answer, captcha_answer)) {
            return bridge.ip_address;
        }

        return "Incorrect answer";
    } 
}
