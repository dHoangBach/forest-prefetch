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

echo "Cleaning ./$1/text/"
rm $1/text/*

