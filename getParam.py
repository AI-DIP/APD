def getMaxDepth(source):
    return source.get_params()['max_depth']

def getCcpAlpha(source):
    return source.get_params()['ccp_alpha']

def getNPrototypes(source):
    proto = source.get_params()['sampling_strategy']
    return proto[0]