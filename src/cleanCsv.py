import csv
import os

def cleanCSV(file, output = 'clean_csv.csv'):
	# Old values from postSporqDraftOutcomes csv
	# valid_columns = ['DP Normalizado', 'Tm # Normalizado', 'Pos # Normalizado', 'Age Normalizado', 'AvgAV Categorizado', 'Conference # Norm', 'NFL.com Grade', 'SPORQ Normalizado']
	valid_columns = ['Pos # Normalizado', 'Age Normalizado', 'AltAAV Cat', 'College # Norm', 'C# Norm', 'Grade', 'SPORQ Normalizado']
	dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
	filename = dir_path + file
	with open(filename) as inf:
		with open(output, 'w', newline='') as csvoutput:
			spamwriter = csv.writer(csvoutput, delimiter=',')
			spamwriter.writerow(valid_columns)
			records = csv.DictReader(inf)
			for row in records:
				new_line = []
				for column in valid_columns:
					if "," in row[column]:
						row[column] = row[column].replace(",", ".")
					new_line.append(row[column])
				spamwriter.writerow(new_line)


# cleanCSV('postSporqDraftOutcomes - postSporqDraftOutcomes.csv')
# cleanCSV('test_inicial.csv', 'test.csv')
# cleanCSV('newAVJoined - newAVJoined.csv', 'newAV_clean.csv')
# cleanCSV('newAVJoinedAltAV1.csv', 'alt_1_clean.csv')
# cleanCSV('newAVJoinedAltAV5.csv', 'alt_5_clean.csv')
cleanCSV('newAVJoinedAltAVSin2018.csv', 'sin_2018_col_noTM_noDP_clean.csv')
cleanCSV('newAVJoinedAltAV2018.csv', 'con_2018_col_noTM_noDP_clean.csv')
