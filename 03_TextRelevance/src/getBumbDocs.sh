#/bin/bash

DIR='../data/data'

find ${DIR} -type d | while read SUBDIRNAME; do	

	if [ "${SUBDIRNAME}" != "${DIR}" ]; then
		continue
	fi
	
	find "${SUBDIRNAME}" -type f | while read FILENAME; do
		echo `cat "$FILENAME" | head -n 1` "$FILENAME" 
	done
done
