from abc import abstractmethod


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math


from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist


from sklearn.model_selection import cross_val_score



class APDBase:
    """
    Class for Prototype Pair Calculations
    It devides the dataset into regions as well as later identify the closes region when making predictions

    """
    def __init__(self, proto, proto_labels, unbalanced_rate:float=0.1, min_support:int=10, prune_regions:bool = True, minimum_n_regions:int =1, metric:str =  'sqeuclidean'):
        """

        :param proto: Prototypes position
        :param proto_labels: Prototypes labels
        :param unbalanced_rate: a rate for aggregating regions it is calculated as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
        :param min_support: minimum number of samples in a single region
        :param prune_regions: if true then the procedure of region pruning would be executed during training
        :param minimum_n_regions: minumum number of regions. If a region do not fulfill the condition it would be aggregated to other region, but only when the minimum number of regions is satisfied
        :param metric: distance metric used in the calculations. by default this is squared euclidien distance. For more details see   scipy.spatial.distance.cdist
        """
        self.ux = None
        if isinstance(proto, pd.DataFrame):
            proto = proto.values
        if isinstance(proto, pd.DataFrame) or isinstance(proto_labels, pd.DataFrame):
            proto_labels = proto_labels.values
        self.proto = proto
        self.proto_labels = proto_labels
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.prune_regions = prune_regions
        self.region_stats = None
        self.minimum_n_regions = minimum_n_regions
        self.metric =  metric

    @staticmethod
    def unpairCantor(z):
        """
        Unpair of the Cantar function.
        Normally for given value it calculates two integer coordinates.
        Here z can be a single value or an array of values, where each
        to each value of z the unpair function will be applied
        :param z: the values to be unpaired
        :return: a tuple of two values or two arrays of values
        """
        t = int(math.floor((math.sqrt(8 * z + 1) - 1) / 2))
        x = int(t * (t + 3) / 2 - z)
        y = int(z - t * (t + 1) / 2)
        return x, y
    @staticmethod
    def _prepare_data(X,y):
        if isinstance(X, pd.DataFrame):  # If dataframe then convert input data to numpy
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values  # If pd.DataFrame then convert input data to numpy
        return X,y

    @staticmethod
    def pairCantor(a, b):
        """
        Pairing function using Cantar formula
        For a given pair of integer values it combines them into a single int value
        Here a and b can be arrays of two integer where each pair of values a[0] b[0] will be paired using Cantar formula
        :param a: value or array of values to be one half of pair
        :param b: value or array of values to be one half of pair
        :return: bired values
        """
        return 0.5 * (a + b) * (a + b + 1) + b

    def _getRegionStats(self, X, y, pairs):
        """
        Function gets as inpyt labeled data, and region assigment and returns and calculates statists over the regions.
        This statistis in a form of a pandas Dataframe is returned where one column is a region_id (can be set as index),
         and the remining columns are: numer of samples in each class in each region (Class1, Class2) and rank which determines the quality
         of a region. The following rank values can be assigned:
         0 - a region contains samples of only one class
         1 - if a region has less samples the defined by the min_support = the minimum number of samples in a region
         2 - if relation between samples of one class over samples from another class is not above a certain threshold:
            np.minimum(s1/s2,s2/s1) < inbalanced_rate - its aim is to keep the number of samples at certain level
        represent statistics including
        :param X: training set
        :param y: labels of the training set
        :param pairs: a dict whos keys are unique_regions id and values are indexes of the elements which belongs to given region
        :return: a DataFrame with statistics for each region
        """
        inbalanced_rate = self.unbalanced_rate
        min_support = self.min_support
        ux_pairs = pairs.keys()
        ux_labels = self.ux

        stats = {"Pair": [],
                 'Class1': [],
                 'Class2': [],
                 'rank': []}
        for pair in ux_pairs:
            id = pairs[pair]
            s1 = np.sum(y[id] == ux_labels[0])
            s2 = np.sum(y[id] == ux_labels[1])
            rank = 100
            if (s1 == 0) or (s2 == 0):
                rank = 0
            else:
                if np.sum(id) < min_support:
                    rank = 1
                else:
                    if np.minimum(s1 / s2, s2 / s1) < inbalanced_rate:
                        rank = 2
            stats['Pair'].append(pair)
            stats['Class1'].append(s1)
            stats['Class2'].append(s2)
            stats['rank'].append(rank)
        stats_df = pd.DataFrame(stats)
        return stats_df

    def _get_most_corrupted_regin(self, stats: pd.DataFrame()):
        stats.sort_values("rank", inplace=True)
        stats.reset_index(drop=True, inplace=True)
        if stats.loc[0, "rank"] < 100:
            return stats.loc[0, "Pair"]
        else:
            return -1

    @abstractmethod
    def assign_regions(self, X:np.ndarray, regions:list|np.ndarray, dist: np.ndarray = None) -> dict:
        """
        For given samples in X it assignes new samples to one of hte regions
        :param X: input data where each row will be assigned to one of existing pairs
        :param regions: a list of unique regions
        :param dist: a matrix of distances between every row in X and every prototype. In None the it will be calculated within the function but it takes alot of time so this matrix can be delivered from outside
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region.
        """
        pass

    @abstractmethod
    def generate_regions(self,
                         X: pd.DataFrame | np.ndarray,
                         y: pd.DataFrame | np.ndarray
                         ) -> dict:
        """
        For input data and already known prototype it generates regions
        :param X:
        :param y:
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region
        """
        pass

class APD(APDBase):
    """
    Class for Prototype Pair Calculations
    It devides the dataset into regions as well as later identify the closes region when making predictions

    """

    def __init__(self, proto, proto_labels, unbalanced_rate:float=0.2, min_support:int=500, prune_regions:bool=True,
                 minimum_n_regions:int=1, metric:str =  'sqeuclidean'):
        """

        :param proto: Prototypes position
        :param proto_labels: Prototypes labels
        :param unbalanced_rate: a rate for aggregating regions it is calculated as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
        :param min_support: minimum number of samples in each region
        :param prune_regions: if true then the procedure of region pruning would be executed during training
        :param minimum_n_regions: minumum number of regions. If a region do not fulfill the condition it would be aggregated to other region, but only when the minimum number of regions is satisfied
        :param metric: distance metric used in the calculations. by default this is squared euclidien distance. For more details see   scipy.spatial.distance.cdist
        """
        super().__init__(proto, proto_labels, unbalanced_rate=unbalanced_rate, min_support= min_support,
                         prune_regions=prune_regions, minimum_n_regions=minimum_n_regions, metric=metric)
        self.regions_inverted_index = {}


    def _check_regions(self,regions):
        if type(regions)==np.ndarray:
            pass
        elif type(regions)==list:
            regions = np.array(regions)
        else:
            raise TypeError("Incorrect type of regions input")
        return regions

    def assign_regions(self, X:np.ndarray, regions: list|np.ndarray, dist: np.ndarray = None) -> dict:
        """
        For given samples in X it assignes new samples to one of hte regions
        :param X: input data where each row will be assigned to one of existing pairs
        :param regions: a list of unique pairs
        :param dist: a matrix of distances between every row in X and every prototype. In None the it will be calculated within the function but it takes alot of time so this matrix can be delivered from outside
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region.
        """
        regions = self._check_regions(regions)

        if dist is None:
            dist = cdist(X, self.proto, metric=self.metric)
        ds = np.zeros((dist.shape[0],
                       regions.shape[0]))  # Allocate memory to store the results - distances to prototypes constituting given pair
        for i,p in enumerate(regions):
            a, b = self.unpairCantor(p)  # Get indexes of prototypes of a pair
            ds[:, i] = dist[:, a] + dist[:,b]  # Get the distance to the pair, note that here i denotes the index of a given pair
        idp = np.argmin(ds, axis=1)  # Find smallest distances ang get index of this nearest pairs
        out = {}
        regins_index = regions[idp]
        for pair in regions:
            out[pair] = regins_index==pair
          # Convert a list of unique pairs to the full array of new pairs
        return out
    
    def get_distance_to_region_prototypes(self, X:np.ndarray, regions:list, dist:np.ndarray = None):
        regions = self._check_regions(regions)

        if dist is None:
            dist = cdist(X, self.proto, metric=self.metric)
        ds = np.zeros((dist.shape[0],
                       regions.shape[0]))  # Allocate memory to store the results - distances to prototypes constituting given pair
        for i,p in enumerate(regions):
            a, b = self.unpairCantor(p)  # Get indexes of prototypes of a pair
            ds[:, i] = dist[:, a] + dist[:,b]  # Get the distance to the pair, note that here i denotes the index of a given pair
        return ds

    def _get_possible_pairs(self, X, y) -> np.ndarray:
        ux = self.ux
        PY = self.proto_labels
        P = self.proto
        # indexes of samples from given class
        idPos = np.squeeze(PY == ux[0])  # Samples from first class
        idNeg = np.squeeze(PY == ux[1])  # Samples from second class
        # for each sample in X it gets nearest samples from both classes
        dist = cdist(X, P, metric=self.metric)
        dPos = dist[:, idPos]
        dNeg = dist[:, idNeg]

        npp = np.argmin(dPos, axis=1)  # Get index of the Nearest prototype positive
        npn = np.argmin(dNeg, axis=1)  # Get index of the Nearest prototype negative

        idPosI = np.nonzero(np.squeeze(idPos))[0]  # Convert binary index into numeric one for positive samples
        idNegI = np.nonzero(np.squeeze(idNeg))[0]  # Convert binary index into numeric one for negative samples

        idPosN = idPosI[npp]
        idNegN = idNegI[npn]

        # Change order so that smaller number is always first to avoid duplicated indees such that from one sample the
        # neares pair is 2, 10, and for the other 10, 2. To avoid it we always start with the smalles index so in both cases that will be 2, 10
        id = idPosN > idNegN
        tmp = idPosN[id]
        idPosN[id] = idNegN[id]
        idNegN[id] = tmp

        # Pairs
        pairs = self.pairCantor(idPosN, idNegN)
        return np.unique(pairs)

    def generate_regions(self, X:pd.DataFrame|np.ndarray, y:pd.DataFrame|np.ndarray) -> dict:
        """
        For input data and already known prototype pairs it assigns samples to given region
        :param X:
        :param y:
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region
        """

        ux = np.unique(self.proto_labels)
        if len(ux) != 2:  # If more then 2 labels then error - the algorithm only supports 2 class problems
            raise ValueError(
                "The algorithm assums binary classification, but the number of prototype classes is != 2")
        self.ux = ux  # Get labels
        X, y = self._prepare_data(X, y)
        ux_pairs = self._get_possible_pairs(X, y)
        dist = cdist(X, self.proto, metric=self.metric)
        self._update_inverted_index(ux_pairs)
        pairs = self._generate_regions_assign(X, ux_pairs, dist)
        stats = self._getRegionStats(X, y, pairs)
        ux_pairs = list(pairs.keys())
        if self.prune_regions:
            # If samples do not fulfill given statisitcs, reasign these samples to one of existing regions
            while ((pair := self._get_most_corrupted_regin(stats)) != -1) and (len(ux_pairs)>self.minimum_n_regions):
                ux_pairs.remove(pair)
                self._update_inverted_index(ux_pairs)
                pairs = self._generate_regions_assign(X, ux_pairs, dist)
                stats = self._getRegionStats(X, y, pairs)
        self.region_stats = stats
        return pairs

    def _generate_regions_assign(self, X: np.ndarray, regions: list | np.ndarray, dist: np.ndarray = None) -> dict:
        return self.assign_regions(X, regions, dist)

    def _update_inverted_index(self, ux_pairs):
        self.regions_inverted_index = {}
        index = self.regions_inverted_index
        for pair in ux_pairs:
            a, b = self.unpairCantor(pair)
            if a not in index: index[a] = []
            if b not in index: index[b] = []
            index[a].append(pair)
            index[b].append(pair)

class APD2(APD):
    """
    Class for Prototype Pair Calculations
    It devides the dataset into regions as well as later identify the closes region when making predictions
    It differes from APD in the way of determining possible regions, here the RNG algorithm is used to determine regions

    """

    def __init__(self, proto, proto_labels, unbalanced_rate:float=0.2, min_support:int=500, prune_regions:bool=True,
                 minimum_n_regions:int=1, metric:str =  'sqeuclidean'):
        """

        :param proto: Prototypes position
        :param proto_labels: Prototypes labels
        :param unbalanced_rate: a rate for aggregating regions it is calculated as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
        :param min_support: minimum number of samples in each region
        :param prune_regions: if true then the procedure of region pruning would be executed during training
        :param minimum_n_regions: minumum number of regions. If a region do not fulfill the condition it would be aggregated to other region, but only when the minimum number of regions is satisfied
        :param metric: distance metric used in the calculations. by default this is squared euclidien distance. For more details see   scipy.spatial.distance.cdist
        """
        super().__init__(proto, proto_labels, unbalanced_rate=unbalanced_rate, min_support= min_support,
                         prune_regions=prune_regions, minimum_n_regions=minimum_n_regions, metric=metric)

    def _get_possible_pairs(self,X,y) -> np.ndarray:
        ux = self.ux
        PY = self.proto_labels
        P = self.proto
        # indexes of samples from given class
        idPos = np.squeeze(PY == ux[0])  # Samples from first class
        idNeg = np.squeeze(PY == ux[1])  # Samples from second class
        dist = cdist(P, P, metric=self.metric)

        idPosI = np.nonzero(np.squeeze(idPos))[0]  # Convert binary index into numeric one for positive samples
        idNegI = np.nonzero(np.squeeze(idNeg))[0]

        n = P.shape[0]

        idPosN = []
        idNegN = []
        for a in idPosI:
            for b in idNegI:
                d_ab = dist[a,b]
                chk = True
                for c in range(n):
                    if (a==c) or (b==c): continue
                    d_ac = dist[a, c]
                    d_bc = dist[b, c]
                    #if np.maximum(d_ac,d_bc) <= d_ab :
                    if d_ac + d_bc <= d_ab:
                        chk = False
                        break
                if chk:
                    idPosN.append(a)
                    idNegN.append(b)

        pairs = self.pairCantor(np.array(idPosN), np.array(idNegN))
        return pairs


class APD3(APD):
    """
    Class for Prototype Pair Calculations
    It devides the dataset into regions as well as later identify the closes region when making predictions
    It differes from APD in the way of determining possible regions, here the RNG algorithm is used to determine regions

    """

    def __init__(self, proto, proto_labels, unbalanced_rate:float=0.2, min_support:int=500, prune_regions:bool=True,
                 minimum_n_regions:int=1, metric:str =  'sqeuclidean'):
        """

        :param proto: Prototypes position
        :param proto_labels: Prototypes labels
        :param unbalanced_rate: a rate for aggregating regions it is calculated as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
        :param min_support: minimum number of samples in each region
        :param prune_regions: if true then the procedure of region pruning would be executed during training
        :param minimum_n_regions: minumum number of regions. If a region do not fulfill the condition it would be aggregated to other region, but only when the minimum number of regions is satisfied
        :param metric: distance metric used in the calculations. by default this is squared euclidien distance. For more details see   scipy.spatial.distance.cdist
        """
        super().__init__(proto, proto_labels, unbalanced_rate=unbalanced_rate, min_support= min_support,
                         prune_regions=prune_regions, minimum_n_regions=minimum_n_regions, metric=metric)

    def _get_possible_pairs(self,X,y) -> np.ndarray:
        ux = self.ux
        PY = self.proto_labels
        P = self.proto
        # indexes of samples from given class
        idPos = np.squeeze(PY == ux[0])  # Samples from first class
        idNeg = np.squeeze(PY == ux[1])  # Samples from second class
        dist = cdist(P, P, metric=self.metric)

        idPosI = np.nonzero(np.squeeze(idPos))[0]  # Convert binary index into numeric one for positive samples
        idNegI = np.nonzero(np.squeeze(idNeg))[0]

        n = P.shape[0]

        idPosN = []
        idNegN = []
        for a in idPosI:
            for b in idNegI:
                d_ab = dist[a,b]
                chk = True
                for c in range(n):
                    if (a==c) or (b==c): continue
                    d_ac = dist[a, c]
                    d_bc = dist[b, c]
                    #if np.maximum(d_ac,d_bc) <= d_ab :
                    if d_ac + d_bc <= d_ab:
                        chk = False
                        break
                if chk:
                    idPosN.append(a)
                    idNegN.append(b)

        pairs = self.pairCantor(np.array(idPosN), np.array(idNegN))
        return pairs

    def _generate_regions_assign(self, X: np.ndarray, regions: list|np.ndarray, dist: np.ndarray = None) -> dict:
        """
        For given samples in X it assignes new samples to one of hte regions during the training call for generate_regions
        :param X: input data where each row will be assigned to one of existing pairs
        :param regions: a list of unique pairs
        :param dist: a matrix of distances between every row in X and every prototype. In None the it will be calculated within the function but it takes alot of time so this matrix can be delivered from outside
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region.
        """
        if type(regions)==np.ndarray:
            pass
        elif type(regions)==list:
            regions = np.array(regions)
        else:
            raise TypeError("Incorrect type of regions input")

        if dist is None:
            dist = cdist(X, self.proto, metric=self.metric)
        protos_id = np.array(sorted(set(sum(map(self.unpairCantor,regions),()))))

        minid = np.argmin(dist[:,protos_id],axis=1)
        minid = protos_id[minid] #Getting index of the nearest prototype
        fun = lambda a,b : (a == minid) | (b == minid)
        out = {region:  fun( *self.unpairCantor(region) ) for region in regions}

        return out

    def assign_regions(self, X: np.ndarray, regions: list|np.ndarray, dist: np.ndarray = None) -> dict:
        """
        For given samples in X it assignes new samples to one of the regions during the prediction phase
        :param X: input data where each row will be assigned to one of existing pairs
        :param regions: a list of unique pairs
        :param dist: a matrix of distances between every row in X and every prototype. In None the it will be calculated within the function but it takes alot of time so this matrix can be delivered from outside
        :return: a dict with keys equal regions id and values equal to samples from X indexes assigned to given region.
        """
        if type(regions)==np.ndarray:
            pass
        elif type(regions)==list:
            regions = np.array(regions)
        else:
            raise TypeError("Incorrect type of regions input")
        index = list(self.regions_inverted_index.keys())
        if dist is None:
            dist = cdist(X, self.proto, metric=self.metric)
        protos_id = np.array(sorted(set(sum(map(self.unpairCantor, regions), ()))))

        minid = np.argmin(dist[:,protos_id],axis=1)
        minid = protos_id[minid] #Getting index of the nearest prototype


        ds = np.zeros((dist.shape[0],
                       regions.shape[0]))  # Allocate memory to store the results - distances to prototypes constituting given pair

        for i,p in enumerate(regions):
            a, b = self.unpairCantor(p)  # Get indexes of prototypes of a pair
            ds[:, i] = dist[:, a] + dist[:,b]# Get the distance to the pair, note that here i denotes the index of a given pair
            chk = (a == minid) | (b == minid)
            ds[~chk,i] = np.inf #If prototype a or b are not in the voronoi cell of the closest prototypes ten set its value to inf. We only want those pairs which are included in the nearest cell

        idp = np.argmin(ds, axis=1)  # Find smallest distances ang get index of this nearest pairs
        out = {}
        regins_index = regions[idp]
        for pair in regions:
            out[pair] = regins_index==pair
          # Convert a list of unique pairs to the full array of new pairs
        return out


class PE(APDBase):
    def __init__(self, proto, proto_labels, unbalanced_rate:float=0.01, min_support:int=10, prune_regions:bool = False,
                 minimum_n_regions:int=1, metric:str =  'sqeuclidean'):
        """
        :param prune_regions: if true then the procedure of region pruning would be executed during training
        :param minimum_n_regions: minumum number of regions. If a region do not fulfill the condition it would be aggregated to other region, but only when the minimum number of regions is satisfied
        :param metric: distance metric used in the calculations. by default this is squared euclidien distance. For more details see   scipy.spatial.distance.cdist
        """
        super().__init__(proto, proto_labels, unbalanced_rate=unbalanced_rate, min_support=min_support,
                         prune_regions=prune_regions, minimum_n_regions=minimum_n_regions, metric=metric )


    def assign_regions(self, X, regions: np.ndarray|list, dist:np.ndarray = None) -> dict:
        """
        For given samples in X it assignes new samples to one of the regions found by generate_regions method
        :param X: input data where each row will be assigned to one of existing pairs
        :param regions: a list of unique pairs
        :param dist: a matrix of distances between every row in X and every prototype. In None the it will be calculated within the function but it takes alot of time so this matrix can be delivered from outside
        :return: the nearest region for each row in X
        """
        if dist is None:
            dist = cdist(X, self.proto[regions, :], metric=self.metric)
        else:
            dist = dist[:, regions]
        idp = np.argmin(dist, axis=1)  # Find smallest distances ang get index of this nearest pairs
        regions = np.array(regions)
        region_index = regions[idp] #Convert a list of unique pairs to the full array of new pairs
        out = {}
        for region in regions:
            out[region] = region_index==region
        return out

    def generate_regions(self,
                         X:pd.DataFrame | np.ndarray, y:pd.DataFrame | np.ndarray) -> dict:
        """
        For input data and already knwon prototype it identifies regions
        :param X:
        :param y:
        :return:
        """
        X, y = self._prepare_data(X, y)
        ux = np.unique(y)  # Get labels
        self.ux = ux
        if len(ux) != 2:  # If more then 2 labels then error - the algorithm only supports 2 class problems
            raise ValueError("The algorithm assums binary classification, but the number of prototype classes is != 2")
        # for each sample in X it gets nearest samples from both classes
        dist = cdist(X, self.proto,
                     metric=self.metric)  # Calculate distance from X to the prototypes from positive class
        sample2region = np.argmin(dist, axis=1)  # Get index of the Nearest prototype positive
        ux_regions = list(np.unique(sample2region))
        regions={}
        for region in ux_regions:
            regions[region] = sample2region==region
        stats = super()._getRegionStats(X, y, regions)
        if self.prune_regions:
            # If samples do not fulfill given statisitcs, reasign these samples to one of existing regions
            while ((pair := super()._get_most_corrupted_regin(stats)) != -1) and (len(ux_regions)>self.minimum_n_regions):
                ux_regions.remove(pair)
                regions = self.assign_regions(X, ux_regions, dist)
                stats = super()._getRegionStats(X, y, regions)
        self.region_stats = stats
        return regions


