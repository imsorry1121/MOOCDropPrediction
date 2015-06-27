from time import time
from datetime import datetime 
fmt = "%Y-%m-%dT%H:%M:%S"
fmt2 = "%Y-%m-%d"

def main(outputFile="../result/time.csv",logFile="../data/train/log_train_short.csv", enrollFile="../data/train/enrollment_train.csv", gtFile="../data/train/truth_train.csv"):
	courseSTime = readCourse()
	gts = readGT(gtFile)
	eids = list()
	eSeries = dict()
	eData = dict()
# enrollment_id,username,course_id,time,source,event,object
# input
	with open(logFile, 'r') as fi:
		titles = fi.readline().strip().split(',')
		for line in fi:
			[eid,uid,cid,time,source,event,obj] = line.strip().split(',')
			t = datetime.strptime(time, fmt)
			tdiff = (t-courseSTime[cid]).days
			if eid not in eids:
				eid = str(eid)
				eids.append(eid)
				eSeries[eid] = [0]*30
				eData[eid] = [uid, cid]
			eSeries[eid][tdiff] = eSeries[eid][tdiff]+1
# output
	with open(outputFile,'w') as fo:
		fo.write("gt,eid,uid,cid"+','.join([str(i) for i in range(30)])+'\n')
		for eid in eids:
			gt = gts[eid]
			outputStr = str(gt)+','+','.join([str(i) for i in (eData[eid]+eSeries[eid])])+'\n'
			fo.write(outputStr)


def readCourse(courseFile="../data/date.csv"):
	courseTime = dict()
	with open(courseFile, 'r') as fi:
		fi.readline()
		for line in fi:
			[cid, start, end] = line.strip().split(",")
			courseTime[cid] = datetime.strptime(start, fmt2)
	return courseTime

def readGT(gtFile):
	enrollGT = dict()
	with open(gtFile, 'r') as fi:
		for line in fi:
			[eid, gt]=line.strip().split(',')
			enrollGT[eid] = gt
	return enrollGT

if __name__=="__main__":
	main()
