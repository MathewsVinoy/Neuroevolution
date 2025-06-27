class Connection:
    """
    Connection:
     This class is used to creat an istance of the connection between 2 nodes.

     self.enable => (dtype = boolen) diable when new node is added then added to the last of the Genome
     innovation numbers =>(dtype = int) unique number reprsinting each connection
    """
    
    def __init__(self,input_id,out_id,weight,enable,innovation_number):
        self.inId = input_id
        self.outId = out_id
        self.weight = weight
        self.enable = enable  
        self.innoNo = innovation_number 

    @staticmethod
    def innnovationTracker(innId, OutId):
        key = (innId, OutId)
        if key not in Config.innovation_tracker:
            Config.innovation_tracker[key] = Config.innovation_no
            Config.innovation_no += 1
        return Config.innovation_tracker[key]