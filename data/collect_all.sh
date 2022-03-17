#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel or ppc)"
  exit 1
fi


#echo "path,filename,mean,variance,min,max,size" > collection_$1.csv

for d in ./*/; do
	if [ "$d" != "./__pycache__/" ]; then
		echo $d/cpp/$1/collection.csv
	
		if [ -f $d/cpp/$1/collection.csv ] ; then
			rm $d/cpp/$1/collection.csv
		fi

		echo "Profiling $d"
		./collect.sh $d $1
		#cat $d/cpp/$1/collection.csv >> collection_$1.csv
	fi
done