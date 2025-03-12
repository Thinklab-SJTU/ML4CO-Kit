import importlib.util

#######################################
#            ATSP Algorithm           #  
#######################################
from .atsp.decoder.greedy import atsp_greedy_decoder
from .atsp.local_search.two_opt import atsp_2opt_local_search

#######################################
#             MCl Algorithm           #  
#######################################
from .mcl.decoder.greedy import mcl_greedy_decoder
from .mcl.decoder.beam import mcl_beam_decoder
from .mcl.decoder.gp_degree import mcl_gp_degree_decoder
from .mcl.decoder.lc_degree import mcl_lc_degree_decoder

#######################################
#            MCut Algorithm           #  
#######################################
from .mcut.decoder.lc_degree import mcut_lc_degree_decoder

#######################################
#             MIS Algorithm           #  
#######################################
from .mis.decoder.greedy import mis_greedy_decoder
from .mis.decoder.beam import mis_beam_decoder
from .mis.decoder.gp_degree import mis_gp_degree_decoder
from .mis.decoder.lc_degree import mis_lc_degree_decoder

#######################################
#             MIS Algorithm           #  
#######################################
from .mvc.decoder.greedy import mvc_greedy_decoder
from .mvc.decoder.gp_degree import mvc_gp_degree_decoder
from .mvc.decoder.lc_degree import mvc_lc_degree_decoder

#######################################
#             TSP Algorithm           #  
#######################################
from .tsp.decoder.mcts import tsp_mcts_decoder
from .tsp.decoder.greedy import tsp_greedy_decoder
from .tsp.decoder.insertion import tsp_insertion_decoder
from .tsp.local_search.mcts import tsp_mcts_local_search

#######################################
#      Extension Function (torch)     #  
#######################################
found_torch = importlib.util.find_spec("torch")
if found_torch is not None:
    from .mcl.decoder.rlsa import mcl_rlsa_decoder
    from .mcl.local_search.rlsa import mcl_rlsa_local_search
    from .mcut.decoder.rlsa import mcut_rlsa_decoder
    from .mcut.local_search.rlsa import mcut_rlsa_local_search
    from .mis.decoder.rlsa import mis_rlsa_decoder
    from .mis.local_search.rlsa import mis_rlsa_local_search
    from .mvc.decoder.rlsa import mvc_rlsa_decoder
    from .mvc.local_search.rlsa import mvc_rlsa_local_search
    from .tsp.local_search.two_opt import tsp_2opt_local_search
