def process(inputFile="../data/test/enrollment_test.csv", outputFile="../result/testSequential.csv", outputFile2="../result/testPattern.csv", courseFile="../data/date.csv"):
	users = list()
	courses = list()
	userCourses = dict()
	# read courses
	with open(courseFile, 'r') as fic:
		fic.readline()
		for line in fic:
			courses.append(line.strip().split(',')[0])
	# read enrollment
	with open(inputFile, 'r') as fi:
		fi.readline()
		for line in fi:
			[eid, uid, cid] = line.strip().split(',')
			if uid not in users:
				users.append(uid)
				userCourses[uid] = [cid]
			else:
				ucourses = userCourses[uid]
				ucourses.append(cid)
				userCourses[uid] = ucourses
	# write sequential
	with open(outputFile, 'w') as fos:
		fos.write("person,course\n")
		for i in range(len(users)):
			user= users[i]
			for course in userCourses[user]:
				fos.write(user+","+course+'\n')
	# write pattern
	with open(outputFile2, 'w') as fop:
		titles = "person,"+','.join(courses)+'\n'
		fop.write(titles)
		for i in range(len(users)):
			user= users[i]
			coursesIndicator = ['?']*len(courses)
			for course in userCourses[user]:
				index = courses.index(course)
				coursesIndicator[index] = '1'
			outputStr = user+','+",".join(coursesIndicator)+'\n'
			fop.write(outputStr)

if __name__ =="__main__":
	process()