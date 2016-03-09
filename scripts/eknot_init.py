import sys,os
import pickle
from eknot_utils import init_all,EventNode
from utils import loadPickle

if __name__ == "__main__":
    # input args: K tweetPre  dataPickle outPickle mini [n_init init_size batch_size]
    K=int(sys.argv[1])
    tweetPre=sys.argv[2]
    [Xs,vects,DT,ind2obj] = loadPickle(sys.argv[3])
    outPickle = sys.argv[4]
    mini = int(sys.argv[5])
    if mini:
        n_init = int(sys.argv[6])
        init_size = int(sys.argv[7])
        batch_size = int(sys.argv[8])
    
    # inits
    sys.stderr.write("begin initiating... \n")
    if mini:
        inits,labels,centers = init_all(K,Xs,DT,mini,n_init,init_size,batch_size)
    else:
        inits,labels,centers = init_all(K,Xs,DT)
    # write
    if outPickle != 'null':
        with open(outPickle, 'w') as f:
            pickle.dump([inits,labels,centers],f)
    
    sys.stderr.write("Pickle saved. Begin printing... \n")
    ####################### print #######################
    rootNode = EventNode(Xs,initsDescriptor=inits)
    rootNode.printCluster(vects,ind2obj,tweetPre=tweetPre,t_topK=1000,fromPlsa=False)
