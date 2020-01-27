#!/bin/sh
die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "USC ID required as argument!"

if [ ${#1} -ne 10 ]
then
    die "USC ID argument should be of length 10!"
fi

rm -f $1.zip 
zip -r $1.zip . -i "Policy_Gradients*.ipynb"
