from backports import configparser

def load(file):
    try:
        config_file = open(file,'r')
    except IOError:
        print("Error file not found", file)
        raise
    else:
        parm = configparser.ConfigParser()
        parm.read_file(config_file)

        Config.input_nodes          =       int(parm.get('phenotype','input_nodes'))
        Config.output_nodes         =       int(parm.get('phenotype','output_nodes'))
        Config.hidden_nodes         =       int(parm.get('phenotype','hidden_nodes'))






class Config:
    input_nodes         = None
    output_nodes        = None
    hidden_nodes        = None
    
    