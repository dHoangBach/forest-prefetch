#/bin/bash

if [ "$#" -lt 1 ]
then
      echo "Please give a (valid) compile target (arm or intel or ppc)"
      exit 1
fi

./init_all.sh
./train_all.sh
echo "Executing $1"
./generate_all.sh $1
./compile_all.sh $1
./collect_all.sh $1
