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

    # $bname file name
    # $(basename $(dirname $d)) last folder/forest ensemble name
    # $(./$bname) output of file (test)
    # $(stat --printf="%s" $bname) size in bytes
for d in $(find ./$1/cpp/$2/*/ -type f -executable -name "test*Tree"); do
      cd $(dirname $d)
      bname=$(basename $d)
      prefix="test"
      suffix="Tree"
      #suffix2="IfTree"
      #suffix3="NativeTree"
      methodname=${bname#$prefix}
      methodname=${methodname%$suffix}
      #methodname=${methodname%$suffix2}
      #methodname=${methodname%$suffix3}

      measurements="$(basename $(dirname $d)),$methodname,$(./$bname),$(stat --printf="%s" $bname)"
      cd ..
      echo $measurements >> collection.csv
      cd ../../../
done
