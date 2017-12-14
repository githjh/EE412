import csv
import sys
import os
import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FILE = "model.npz"

NUM_USERS = 943
NUM_ITEMS = 1682
NUM_TOP = 10
TS_MAX = 893286638 # Typically current unix timestamp

class RatingPredictor(object):
	def __init__(self, num_users, num_items):
		self._num_users = num_users
		self._num_items = num_items
		self._R = np.zeros((num_users, num_items))
		self._T = np.zeros((num_users, num_items))
		self._D_users = np.zeros((num_users, num_users))
		self._D_itmes = np.zeros((num_items, num_items))
		self._min_ts = TS_MAX

	def save(self, sfile):
		np.savez(sfile, R=self._R, T=self._T, D_item=self._D_item, D_user=self._D_user, min_ts=self._min_ts)

	def load(self, lfile):
		if not os.path.isfile(lfile):
			print("ERROR: you need to train first.")
			print("TRY 'python %s train TRAIN_FILE_NAME'" % sys.argv[0])
			exit(1)

		npzfile = np.load(lfile)
		self._R = npzfile['R']
		self._T = npzfile['T']
		self._D_user = npzfile['D_user']
		self._D_item = npzfile['D_item']
		self._min_ts = npzfile['min_ts']

	def timef(self, ts1, ts2):
		t_diff = abs(ts1 - ts2)
		return np.exp(2*np.log(0.5)*t_diff)

	def train(self, tfile):
		data_list = []
		with open(tfile) as f:
			data_reader = csv.reader(f, delimiter="\t")
			data_list = list(data_reader)

		for i in range(len(data_list)):
			row = data_list[i]
			idx_u = int(row[0])-1
			idx_i = int(row[1])-1
			rate = float(row[2])
			ts = int(row[3])
			self._R[idx_u][idx_i] = rate
			self._T[idx_u][idx_i] = ts

			if ts > 0 and (ts < self._min_ts):
				self._min_ts = ts

		self._T = np.maximum((self._T - self._min_ts)/(TS_MAX-self._min_ts),0)
		self._D_item = cosine_similarity(np.matrix.transpose(self._R),np.matrix.transpose(self._R))
		self._D_user = cosine_similarity(self._R,self._R)

	def predict(self, user, item, ts):
		if (self._R[user][item] != 0):
			return self._R[user][item]

		d_i = self._D_item[item][:]

		item_idx_list = [x for x in range(NUM_ITEMS) if (d_i[x] != 0.0 and d_i[x] != 1.0 and self._R[user][x] > 0.0)]

		item_idx_list = sorted(item_idx_list, key=lambda i: abs(d_i[i]), reverse=True)
		item_idx_list = item_idx_list[:min(len(item_idx_list)-1,NUM_TOP)]

		d_u = self._D_user[user][:]
		user_idx_list = [x for x in range(NUM_USERS) if (d_u[x] != 0.0 and d_u[x] != 1.0 and self._R[x][item] > 0.0)]

		user_idx_list = sorted(user_idx_list, key=lambda i: abs(d_u[i]), reverse=True)
		user_idx_list = user_idx_list[:min(len(user_idx_list)-1,NUM_TOP)]

		if (len(user_idx_list) + len(item_idx_list) == 0): # impossible to guess
			return 0

		nom = 0.0
		denom = 0.0
		for i in item_idx_list:
			weight = d_i[i] * self.timef(self._T[user][i], float(ts-self._min_ts)/float(TS_MAX-self._min_ts))
			nom += (weight * self._R[user][i])
			denom += abs(weight)

		for u in user_idx_list:
			weight = d_u[u] * self.timef(self._T[u][item], float(ts-self._min_ts)/float(TS_MAX-self._min_ts))
			nom += (weight * self._R[u][item])
			denom += abs(weight)

		ret = float(nom)/float(denom)

		if ret < 1.0:
			ret = 1.0
		elif ret > 5.0:
			ret = 5.0

		return ret

	def validate(self, fin):
		with open(fin) as fi:
			data_reader = csv.reader(fi, delimiter="\t")
			data_list = list(data_reader)

		c = len(data_list)
		err_cum = 0.0
		sys.stdout.write('\r' + "Evaluating... (0/%d)" % c)
		for i in range(len(data_list)):
			sys.stdout.write('\r' + "Evaluating... (%d/%d)" % (i,c))
			row = data_list[i]
			if len(row) != 4:
				print("File format error. Evaluation file should be tab-separated.")
				exit(1)
			idx_u = int(row[0])-1
			idx_i = int(row[1])-1
			trate = int(row[2])
			ts = int(row[3])

			R_e = self.predict(idx_u, idx_i, ts)
			err_cum += (trate - R_e)**2
		
		print("\nRMSE = %f" % np.sqrt(err_cum/float(c)))

	def evaluate(self, fin, fout):
		with open(fin,'r') as fi:
			data_reader = csv.reader(fi,delimiter=",")
			data_list = list(data_reader)

		c = len(data_list)

		with open(fout,'w') as fo:
			data_writer = csv.writer(fo)
			sys.stdout.write('\r' + "Testing... (0/%d)" % c)
			for i in range(len(data_list)):
				sys.stdout.write('\r' + "Testing... (%d/%d)" % (i,c))
				row = data_list[i]
				if len(row) != 3:
					print("File format eror. Test file should be comma-separated.")
					exit(1)
				idx_u = int(row[0])-1
				idx_i = int(row[1])-1
				ts = int(row[2])
				data_writer.writerow([idx_u+1,idx_i+1,self.predict(idx_u,idx_i,ts)])
			print("\nOutput is written to '%s'" % fout);	

def clean():
	if os.path.isfile(MODEL_FILE):
		os.remove(MODEL_FILE)

def printUsage():
	print("<Usage>")
	print("  For training:")
	print("\tpython %s train TRAIN_FILE_NAME" % sys.argv[0])
	print("  For validation:")
	print("\tpython %s validate EVAL_FILE_NAME" % sys.argv[0])
	print("  For evaluate:")
	print("\tpython %s evaluate EVAL_FILE_NAME" % sys.argv[0])
	print("  To clean up:")
	print("\tpython %s clean" % sys.argv[0])

def main(argv):	
	if len(argv) == 3 and argv[1] == 'train':
		r = RatingPredictor(NUM_USERS, NUM_ITEMS)
		r.train(argv[2])
		r.save(MODEL_FILE)
	elif len(argv) == 3 and argv[1] == 'validate':
		r = RatingPredictor(NUM_USERS, NUM_ITEMS)
		r.load(MODEL_FILE)
		r.validate(argv[2])
	elif len(argv) == 4 and argv[1] == 'evaluate':
		r = RatingPredictor(NUM_USERS, NUM_ITEMS)
		r.load(MODEL_FILE)
		r.evaluate(argv[2],argv[3])
	elif len(argv) == 2 and argv[1] == 'clean':
		clean();
	else:
		printUsage()

if __name__ == '__main__':
	main(sys.argv)
