class SearchConfigurator:
    """
    SearchConfigurator class is used to configure the decoding and
    local search functions for the search algorithms.
    """

    def __init__(self, decoding_class, local_search_class):
        self.decoding_class = decoding_class  # decoding functions
        self.local_search_class = local_search_class  # local search functions

    def get_decoding_func(self, task, decoding_type):
        return self.decoding_class(task, decoding_type)

    def get_local_search_func(self, task, local_search_type):
        return self.decoding_class(task, local_search_type)
