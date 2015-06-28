from time import time
from datetime import datetime 
fmt = "%Y-%m-%dT%H:%M:%S"
fmt2 = "%Y-%m-%d"

def main(outputFile="../result/time.csv",logFile="../data/train/log_train_short.csv", enrollFile="../data/train/enrollment_train.csv", gtFile="../data/train/truth_train.csv"):
	courseSTime = readCourseTime()
	gts = readGT(gtFile)
	eData = readEnrollData(enrollFile)
	outputTimeSeries(courseSTime, gts, eData, logFile, outputFile)

	# enrollment_id,username,course_id,time,source,event,object
	# input
	



def outputTimeSeries(courseSTime, gts, eData, logFile, outputFile):
	eids = list()
	eSeries = dict()
	with open(logFile, 'r') as fi:
		titles = fi.readline().strip().split(',')
		for line in fi:
			[eid,time,source,event,obj] = line.strip().split(',')
			t = datetime.strptime(time, fmt)
			cid = eData[eid][1]
			tdiff = (t-courseSTime[cid]).days
			if eid not in eids:
				eid = str(eid)
				eids.append(eid)
				eSeries[eid] = [0]*30
			eSeries[eid][tdiff] = eSeries[eid][tdiff]+1
	# output
	outputFileDrop = "../result/timeDrop.csv"
	outputFileRemain = "../result/timeRemain.csv"
	# outputFileRatioDrop = "../result/timeRatioDrop.csv"
	# outputFileRatioRemain = "../result/timeRatioRemain.csv"
	fo2 = open(outputFileDrop, 'w')
	fo3 = open(outputFileRemain, 'w')
	with open(outputFile,'w') as fo:
		fo.write("gt,eid,uid,cid,"+','.join([str(i) for i in range(30)])+'\n')
		for eid in eids:
			gt = gts[eid]
			outputStr = str(gt)+','+eid+','+','.join([str(i) for i in (eData[eid]+eSeries[eid])])+'\n'
			fo.write(outputStr)
			if gt == '0':
				fo3.write(outputStr)
			else:
				fo2.write(outputStr)
	fo2.close()
	fo3.close()


def readCourseTime(courseFile="../data/date.csv"):
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

def readEnrollData(enrollFile):
	eData = dict()
	with open(enrollFile, 'r') as fi:
		fi.readline()
		for line in fi:
			[eid, uid, cid] = line.strip().split(',')
			eData[eid] = [uid, cid]
	return eData


if __name__=="__main__":
	# short 
	# main()

	# training
	# main(outputFile="../result/time.csv",logFile="../data/train/log_train.csv", enrollFile="../data/train/enrollment_train.csv", gtFile="../data/train/truth_train.csv"):
	# testing 
	main(outputFile="../result/timeTest.csv",logFile="../data/test/log_test.csv", enrollFile="../data/test/enrollment_test.csv", gtFile="../data/test/origin_test.csv")
