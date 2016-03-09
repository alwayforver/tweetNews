import sys,os
import pickle
from eknot_utils import nextData,weightX,subRun,EventNode,NodeParams
from utils import loadPickle

if __name__ == "__main__":
    # data_pickle plsa_pickle tweetPre switchText eventID Kevent [outPickle]
    [Xs,vects,DT,ind2obj] = loadPickle(sys.argv[1])
    rootParams,rootNodeDescriptor = loadPickle(sys.argv[2])
    tweetPre = sys.argv[3]
    switch = sys.argv[4]
    eventID = int(sys.argv[5]) # event number
    K=int(sys.argv[6])
    if len(sys.argv)>7:
        outPickle = sys.argv[7]

    rootNode = EventNode(Xs,params=rootParams,descriptor=rootNodeDescriptor)

    ######### sub ###########
    sys.stderr.write('Running sub...\n')
    n_wdxPz_wds,XsWeighted = nextData(rootNode)
    ## params
    numX = len(XsWeighted)
    Xinds = range(numX)  # can be customized
    ws = [1 for i in Xinds] # can be customized
    selectTime = 0  #
    wt = 0.5
    lambdaB = 0.5
    Learn=(1,10)
    ##
    params = NodeParams(Xinds,ws,lambdaB,selectTime,wt,Learn,eventID)
    eventNode = subRun(XsWeighted,n_wdxPz_wds,K,params,DT)
    ## out pickle
    if len(sys.argv)>7:
        sys.stderr.write('saving event node pickle...\n')
        with open(outPickle, 'w') as f:
            pickle.dump(eventNode,f)          
    ## print
    eventNode.printCluster(vects,ind2obj,tweetPre=tweetPre)
