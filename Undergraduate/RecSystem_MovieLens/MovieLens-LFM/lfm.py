# -*- coding = utf-8 -*-
import collections
import itertools
import math
import os
import pickle
import random
import shutil
from collections import defaultdict
from operator import itemgetter

random.seed(0)

class ModelManager:
    """
    Model manager is designed to load and save all models.
    No matter what dataset name.
    """
    path_name = ''

    @classmethod
    def __init__(cls, dataset_name="", test_size=0.3):
        """
        cls.dataset_name should only init for only once.
        :param dataset_name:
        """
        if not cls.path_name:
            cls.path_name = "model/" + dataset_name + '-testsize' + str(test_size)

    def save_model(self, model, save_name: str):
        """
        Save model to model/ dir.
        :param model: source model
        :param save_name: model saved name.
        :return: None
        """
        if 'pkl' not in save_name:
            save_name += '.pkl'
        if not os.path.exists('model'):
            os.mkdir('model')
        pickle.dump(model, open(self.path_name + "-%s" % save_name, "wb"))

    def load_model(self, model_name: str):
        """
        Load model from model/ dir via model name.
        :param model_name:
        :return: loaded model
        """
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(self.path_name + "-%s" % model_name):
            raise OSError('There is no model named %s in model/ dir' % model_name)
        return pickle.load(open(self.path_name + "-%s" % model_name, "rb"))

    @staticmethod
    def clean_workspace(clean=False):
        """
        Clean the whole workspace.
        All File in model/ dir will be removed.
        :param clean: Boolean. Clean workspace or not.
        :return: None
        """
        if clean and os.path.exists('model'):
            shutil.rmtree('model')

class LFM:
    """
    Latent Factor Model.
    Top-N recommendation.
    """

    def __init__(self, K, epochs, alpha, lamb, n_rec_movie=10, save_model=True):
        """
        Init LFM with K, T, alpha, lamb
        :param K: Latent Factor dimension
        :param epochs: epochs to go
        :param alpha: study rate
        :param lamb: regular params
        :param save_model: save model
        """
        print("LFM start...\n")
        self.K = K
        self.epochs = epochs
        self.alpha = alpha
        self.lamb = lamb
        self.n_rec_movie = n_rec_movie
        self.save_model = save_model
        self.users_set, self.items_set = set(), set()
        self.items_list = list()
        self.P, self.Q = None, None
        self.trainset = None
        self.testset = None
        self.item_popular, self.items_count = None, None
        self.model_name = 'K={}-epochs={}-alpha={}-lamb={}'.format(self.K, self.epochs, self.alpha, self.lamb)

    def init_model(self, users_set, items_set, K):
        """
        Init model, set P and Q with random numbers.
        :param users_set: Users set
        :param items_set: Items set
        :param K: Latent factor dimension.
        :return: None
        """
        self.P = dict()
        self.Q = dict()
        for user in users_set:
            self.P[user] = [random.random() for _ in range(K)]
        for item in items_set:
            self.Q[item] = [random.random() for _ in range(K)]

    def init_users_items_set(self, trainset):
        """
        Get users set and items set.
        :param trainset: train dataset
        :return: Basic users and items set, etc.
        """
        users_set, items_set = set(), set()
        items_list = []
        item_popular = defaultdict(int)
        for user, movies in trainset.items():
            for item in movies:
                item_popular[item] += 1
                users_set.add(user)
                items_set.add(item)
                items_list.append(item)
        items_count = len(items_set)
        return users_set, items_set, items_list, item_popular, items_count

    def gen_negative_sample(self, items: dict):
        """
        Generate negative samples
        :param items: Original items, positive sample
        :return: Positive and negative samples
        """
        samples = dict()
        for item, rate in items.items():
            samples[item] = 1
        for i in range(len(items) * 322):
            item = self.items_list[random.randint(0, len(self.items_list) - 1)]
            if item in samples:
                continue
            samples[item] = 0
            if len(samples) >= len(items) * 320:
                break
        #print(len(samples))
        return samples

    def predict(self, user, item):
        """
        Predict the rate for item given user and P and Q.
        :param user: Given a user
        :param item: Given a item to predict the rate
        :return: The predict rate
        """
        rate_e = 0
        for k in range(self.K):
            Puk = self.P[user][k]
            Qki = self.Q[item][k]
            rate_e += Puk * Qki
        return rate_e

    def train(self, trainset):
        """
        Train model.
        :param trainset: Origin trainset.
        :return: None
        """
        for epoch in range(self.epochs):
            print('epoch:', epoch)
            for user in trainset:
                samples = self.gen_negative_sample(trainset[user])
                for item, rui in samples.items():
                    eui = rui - self.predict(user, item)
                    for k in range(self.K):
                        self.P[user][k] += self.alpha * (eui * self.Q[item][k] - self.lamb * self.P[user][k])
                        self.Q[item][k] += self.alpha * (eui * self.P[user][k] - self.lamb * self.Q[item][k])
            self.alpha *= 0.9
            # print(self.P)
            # print(self.Q)

    def fit(self, trainset):
        """
        Fit the trainset by optimize the P and Q.
        :param trainset: train dataset
        :return: None
        """
        self.trainset = trainset
        self.users_set, self.items_set, self.items_list, self.item_popular, self.items_count = \
            self.init_users_items_set(trainset)
        model_manager = ModelManager()
        try:
            self.P = model_manager.load_model(self.model_name + '-P')
            self.Q = model_manager.load_model(self.model_name + '-Q')
            print('User origin similarity model has saved before.\nLoad model succ...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.init_model(self.users_set, self.items_set, self.K)
            self.train(self.trainset)
            print('Train a new model succ.')
            if self.save_model:
                model_manager.save_model(self.P, self.model_name + '-P')
                model_manager.save_model(self.Q, self.model_name + '-Q')
            print('The new model has saved succ...\n')
        return self.P, self.Q

    def recommend(self, user):
        """
        Recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        rank = collections.defaultdict(float)
        interacted_items = self.trainset[user]
        for item in self.items_set:
            if item in interacted_items.keys():
                continue
            for k, Qik in enumerate(self.Q[item]):
                rank[item] += self.P[user][k] * Qik
        return [movie for movie, _ in sorted(rank.items(), key=itemgetter(1), reverse=True)][:self.n_rec_movie]

    def test(self, testset):
        """
        Test the recommendation system by recommending scores to all users in testset.
        :param testset: test dataset
        :return: None
        """
        self.testset = testset
        print('Test recommendation system start...')
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # record the calculate time has spent.
        #test_time = LogTime(print_step=1000)
        for user in self.users_set:
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie in rec_movies:
                if movie in test_movies.keys():
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.item_popular[movie])
            rec_count += self.n_rec_movie
            test_count += len(test_movies)
            #test_time.count_time()
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.items_count)
        popularity = popular_sum / (1.0 * rec_count)
        print('Test recommendation system success.')
        #test_time.finish()
        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))


def parse_line(line: str, sep: str):
        user, movie, rate = line.strip('\r\n').split(sep)[:3]
        return user, movie, rate

def load_dataset(name='ml-1m'):
        if not os.path.isfile(path):
            raise OSError(
                "Dataset data/" + name + " could not be found in this project.\n")
        with open(path) as f:
            ratings = [parse_line(line, sep) for line in itertools.islice(f, 0, None)]
        print("Load " + name + " dataset success.")
        return ratings
        
def train_test_split(ratings, test_size=0.2):
        """
        Split rating data to training set and test set.

        The default `test_size` is the test percentage of test size.

        The rating file should be a instance of DataSet.

        :param ratings: raw dataset
        :param test_size: the percentage of test size.
        :return: train_set and test_set
        """
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainset_len = 0
        testset_len = 0
        for user, movie, rate in ratings:
            if random.random() <= test_size:
                test[user][movie] = int(rate)
                testset_len += 1
            else:
                train[user][movie] = int(rate)
                trainset_len += 1
        print('split rating data to training set and test set success.')
        print('train set size = %s' % trainset_len)
        print('test set size = %s\n' % testset_len)
        return train, test

def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()

if __name__ == '__main__':
    dataset_name = 'ml-100k'
    model_type = 'LFM'
    test_size = 0.3
    #path='data/ml-1m/ratings.dat'
    path='data/ml-100k/u.data'
    #sep='::'
    sep='\t'
    ratings = load_dataset(name=dataset_name)
    trainset, testset = train_test_split(ratings, test_size=test_size)
    model = LFM(20, 10, 0.07, 0.02, 10)
    model.fit(trainset)
    recommend_test(model, [234])
    model.test(testset)