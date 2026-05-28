**Graph: MCl & MCut & MIS & MVC; ✔: Supported; 📆: Planned for future versions (contributions welcomed!).**


| Wrapper | TXT | Other R&W |
| :-----: | --- | :-------: |
| **Routing Tasks** |
| ATSPWrapper               | "[dists] output [sol]" | ``tsplib`` |
| CVRPWrapper               | "depots [depots] points [points] demands [demands] capacity [capacity] output [sol]" | ``vrplib`` |
| CVRPBWrapper              | "depots [depots] points [points] demands [demands] capacity [capacity] output [sol]" | |
| CVRPBLWrapper             | "depots [depots] points [points] demands [demands] capacity [capacity] max_route_length [max_route_length] output [sol]" | |
| CVRPBLTWWrapper           | "depots [depots] points [points] demands [demands] capacity [capacity] tw [tw] service [service] max_route_length [max_route_length] output [sol]" | |
| CVRPBTWWrapper            | "depots [depots] points [points] demands [demands] capacity [capacity] tw [tw] service [service] output [sol]" | |
| CVRPLWrapper              | "depots [depots] points [points] demands [demands] capacity [capacity] max_route_length [max_route_length] output [sol]" | |
| CVRPLTWWrapper            | "depots [depots] points [points] demands [demands] capacity [capacity] tw [tw] service [service] max_route_length [max_route_length] output [sol]" | |
| CVRPTWWrapper             | "depots [depots] points [points] demands [demands] capacity [capacity] tw [tw] service [service] output [sol]" | |
| ORWrapper                 | "depots [depots] points [points] prizes [prizes] max_length [max_length] output [sol]" | |
| PCTSPWrapper              | "depots [depots] points [points] penalties [penalties] prizes [prizes] required_prize [required_prize] output [sol]" | |
| SPCTSPWrapper             | "depots [depots] points [points] penalties [penalties] expected_prizes [expected_prizes] actual_prizes [actual_prizes] required_prize [required_prize] output [sol]" | |
| TSPWrapper                | "[points] output [sol]" | ``tsplib`` |
| **Graph Tasks** |
| (Graph)Wrapper            | "[edge_index] label [sol]" | ``gpickle`` |
| (Graph)Wrapper [weighted] | "[edge_index] weights [weights] label [sol]" | ``gpickle`` |
| **QAP Tasks** |
| GMWrapper                 | -- | ``pickle`` |
| GEDWrapper                | -- | ``pickle`` |
| KQAPWrapper               | -- | ``pickle`` |
| **SAT Tasks** |
| SATPWrapper               | "[vars_num] vars_num [clauses] output [sol]" | ``cnf`` |
| SATAWrapper               | "[vars_num] vars_num [clauses] output [sol]" | ``cnf`` |
| **Portfolio Tasks** |
| MaxRetPOWrapper           | "[returns] cov [cov] max_var [max_var] output [sol]" | |
| MinVarPOWrapper           | "[returns] cov [cov] required_returns [required_returns] output [sol]" | |
| MOPOWrapper               | "[returns] cov [cov] var_factor [var_factor] output [sol]" | |
