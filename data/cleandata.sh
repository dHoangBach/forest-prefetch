#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

if [ "$#" -lt 2 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

echo "Cleaning ./$1/cpp/$2"
rm -r $1/cpp/$2

# this script will only clean the data created by scripts. It does not remove the trained models