pragma solidity >=0.4.21 <0.6.0;

contract Ponte {

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
    function update_bridges(string memory new_bridges) public view {
        require(msg.sender == owner, "Must be owner");
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
        if (compare_strings(bridge.captcha, captcha_answer)) {
            return bridge.ip_address;
        }

        return "Incorrect answer";
    } 
}
