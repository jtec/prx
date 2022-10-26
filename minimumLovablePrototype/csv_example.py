import csv

import example_observations

single_observation_dict = example_observations.generate()
# write single epoch to csv file
with open('example.csv', 'w',newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(single_observation_dict))
    writer.writeheader()
    writer.writerow(single_observation_dict)
