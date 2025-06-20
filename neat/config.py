class Config:
    innovation_no = 0
    node_no = 0
    compatibility_threshold = 3.0
    innovation_tracker ={}
    def __init__(self):
        pass
    
    @staticmethod
    def innnovationTracker(innId, OutId):
        key = (innId, OutId)
        if key not in Config.innovation_tracker:
            Config.innovation_tracker[key] = Config.innovation_no
            Config.innovation_no += 1
        return Config.innovation_tracker[key]