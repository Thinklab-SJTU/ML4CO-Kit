**Graph: MCl & MCut & MIS & MVC; ✔: Supported; 📆: Planned for future versions (contributions welcomed!).**


| Wrapper | TXT | Other R&W |
| :-----: | --- | :-------: |
| ATSPWrapper               | "[dists] output [sol]" | ``tsplib`` |
| CVRPWrapper               | "depots [depots] points [points] demands [demands] capacity [capacity] output [sol]" | ``vrplib`` |
| ORWrapper                 | "depots [depots] points [points] prizes [prizes] max_length [max_length] output [sol]" | |
| PCTSPWrapper              | "depots [depots] points [points] penalties [penalties] prizes [prizes] required_prize [required_prize] output [sol]" | |
| SPCTSPWrapper             | "depots [depots] points [points] penalties [penalties] expected_prizes [expected_prizes] actual_prizes [actual_prizes] required_prize [required_prize] output [sol]" | |
| TSPWrapper                | "[points] output [sol]" | ``tsplib`` |
| (Graph)Wrapper            | "[edge_index] label [sol]" | ``gpickle`` |
| (Graph)Wrapper [weighted] | "[edge_index] weights [weights] label [sol]" | ``gpickle`` |
| MaxRetPOWrapper           | "[returns] cov [cov] max_var [max_var] output [sol]" | |
| MinVarPOWrapper           | "[returns] cov [cov] required_returns [required_returns] output [sol]" | |
| MOPOWrapper               | "[returns] cov [cov] var_factor [var_factor] output [sol]" | |
