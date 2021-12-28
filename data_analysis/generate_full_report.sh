#!/bin/bash

export OUTPUT="all_dataset_report.txt"
export HEAD="================================================================================================================================================================================"


rm $OUTPUT 2>/dev/null

for file in $(ls *txt); do 
	echo "$HEAD">>${OUTPUT}; 
	echo $file>>${OUTPUT};
	echo "$HEAD">>${OUTPUT}; 
	cat $file >>${OUTPUT}; 
	echo -e "\n\n\n\n\n\n" >>${OUTPUT}; 
done
