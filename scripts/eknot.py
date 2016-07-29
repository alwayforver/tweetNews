import sys,os
import pickle
from eknot_utils import EventNode,NodeParams
from utils import loadPickle

def main():
    [Xs,vects,DT,ind2obj] = loadPickle(sys.argv[1])
    [inits,labels,centers] = loadPickle(sys.argv[2])
    tweetPre=sys.argv[3]
    if len(sys.argv)>4:
        outPickle = sys.argv[4]

   ### params move to argv ### 
    eventID=0
#    Xinds = [0,1,2,3]
    Xinds = [0,5]
    ww = 1
    wp=1
    wl=1
    wo=1
#    ws = [ww,wp,wl,wo]
    ws = [ww,1]
    selectTime = 1  #
    wt = 0.5
    lambdaB = 0.5
    Learn=(1,10)  #
    params = NodeParams(Xinds,ws,lambdaB,selectTime,wt,Learn)
    ### run ###
    rootNode = EventNode(Xs,DT,params,inits)
    rootNode.run()
    ########## pickle #######
    if len(sys.argv)>4:
        sys.stderr.write('saving pickle...\n')
        with open(outPickle, 'w') as f:
            pickle.dump([params,rootNode.descriptor],f)          
    ########### print #######
    sys.stderr.write('After pickle, printing...\n')
    if len(sys.argv)>5:
        rootNode.printCluster(vects,ind2obj,tweetPre=tweetPre)

# 4 argv
if __name__ == "__main__":
    # data_pickle inits_pickle tweetPre out_plsa_pickle PRINT
    main()
