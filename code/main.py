import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def parse_arguments():

    parser = argparse.ArgumentParser(description="Run data analysis.")
    parser.add_argument('--input_path', type = str, action="store", dest="input_path", help='Input graph path')

    parser.add_argument('--embedding_size', type = int, action="store", dest="embedding_size", help='Embedding size')
    parser.add_argument('--max_it', type = int, default= 100, action="store", dest="max_it", help='Maximum amount of iterations')

    parser.add_argument('--output_path', type = str, action="store", dest="output_path", help='Output embedding file path')

    return parser.parse_args()



class SirGN:
    def __init__(self, st,embsize=50,maxIter=1000,sep=' '):
        self.embsize=embsize
        self.maxIter=maxIter

        self.nodes=dict()
        self.outedge=dict()
        self.cont=0

        with open(st, "r", encoding = "utf-8") as c:
            for l in c:
                a=l.split(sep)

                b=a[0].replace('\n','')
                a=a[1].replace('\n','')

                if not a in self.nodes:
                    self.nodes[a]=self.cont
                    self.cont=self.cont+1

                    if not self.nodes[a] in self.outedge:
                        self.outedge[self.nodes[a]]=set()

                if not b in self.nodes:
                    self.nodes[b]=self.cont
                    self.cont=self.cont+1

                    if not self.nodes[b] in self.outedge:
                        self.outedge[self.nodes[b]]=set()


        with open(st, "r", encoding = "utf-8") as c:
            for l in c:
                a=l.split(sep)
                b=a[1].replace('\n','')
                a=a[0].replace('\n','')

                self.outedge[self.nodes[a]].add(self.nodes[b])
                self.outedge[self.nodes[b]].add(self.nodes[a])



        self.sol=self.initialize()



    def initialize(self):

        dimr=[len(self.outedge[x]) for x in range(self.cont)]

        print('Node ammount: {}'.format(len(dimr)))

        dim_list = list(set(dimr))
        dim_list.sort()
        dist_dim = len(dim_list)
        dim_to_pos = dict(zip(dim_list,range(dist_dim)))

        print('Objective dimension: {}'.format(self.embsize))

        if self.embsize > len(dimr):
            print('Objective dimension bigger that node ammount')
            print('New objective dimension: {}'.format(len(dimr)))
            self.embsize = len(dimr)


        ret = np.zeros((self.cont,self.embsize))

        for i,d in enumerate(dimr):
            ret[i,0]=d

        self.cluster_labels = [dim_to_pos[x] for x in dimr]

        return ret

    def learn(self):

        it1=0
        scaler = MinMaxScaler()
        self.sol = scaler.fit_transform(self.sol)

        while it1<self.maxIter:
            print('iteration: ',it1+1,' ------------------------------------------------------')

            scaler = MinMaxScaler()
            self.sol = scaler.fit_transform(self.sol)

            cluster_alg = KMeans(n_clusters=self.sol.shape[1],random_state=100)

            self.cluster_labels = cluster_alg.fit_predict(self.sol)

            used_set = set(self.cluster_labels)
            dist_mat = np.zeros_like(self.sol)
            for p1 in range(self.sol.shape[0]):
                max_dist= 0
                min_dist= 1000000000
                dist_dict= {}
                for p2 in range(cluster_alg.cluster_centers_.shape[0]):
                    if p2 in used_set:
                        current_dist = np.linalg.norm(self.sol[p1,:]-cluster_alg.cluster_centers_[p2,:])
                        dist_dict[p2] = current_dist

                        if current_dist > max_dist:
                            max_dist = current_dist

                        if current_dist < min_dist:
                            min_dist = current_dist

                for p2 in range(cluster_alg.cluster_centers_.shape[0]):
                    if p2 in used_set:
                        dist_mat[p1,p2] = (max_dist-dist_dict[p2])/(max_dist-min_dist)
                    else:
                        dist_mat[p1,p2] = 0

            for i_ in range(self.sol.shape[0]):
                dist_mat[i_,:]/=np.sum(dist_mat[i_,:])


            # generate emb
            ret = np.zeros_like(self.sol)
            for i in range(self.sol.shape[0]):
                for neigh_id in self.outedge[i]:
                    ret[i,:]+=dist_mat[neigh_id,:]


            self.sol= ret

            it1+=1




    def write(self,f):
        if np.max(self.cluster_labels) < self.embsize:
            print('Unused dimension number: {}'.format(self.embsize-(np.max(self.cluster_labels)+1)))

        out_sol = self.sol

        print('Storing size {}'.format(out_sol.shape[1]))
        file1 = open(f,'w')
        for h in self.nodes:
            l=self.nodes[h]
            v=''
            ll=out_sol[l,:]
            for g in range(ll.shape[0]):
                v+=' '+str(ll[g])
            file1.write(h+v+'\n')
        file1.close()


args = parse_arguments()

p = SirGN(args.input_path,embsize=args.embedding_size,maxIter=args.max_it)

p.learn()
p.write(args.output_path)
