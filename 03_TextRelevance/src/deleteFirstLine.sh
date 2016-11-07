#/bin/bash

DIR='../data/data'
DIR2='../data/data2'

ls ${DIR} | while read SUBDIRNAME; do
	echo "${DIR}/${SUBDIRNAME}"	
	mkdir "${DIR2}/${SUBDIRNAME}"
	ls "${DIR}/${SUBDIRNAME}" | while read FILENAME; do
		sed -e '1d' "${DIR}/${SUBDIRNAME}/${FILENAME}" > "${DIR2}/${SUBDIRNAME}/${FILENAME}"
	done
done
