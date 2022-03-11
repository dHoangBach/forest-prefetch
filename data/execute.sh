#/bin/bash

if [ "$#" -lt 1 ]
then
      echo "Please give a (valid) sub-folder"
      exit 1
fi

if [ "$#" -lt 2 ]
then
      echo "Please give a (valid) compile target (arm or intel or ppc)"
      exit 1
fi

echo "Executing $1 $2"
./generatePrefetched.py $1 $2
./compile.sh $1 $2
./collect.sh $1 $2
