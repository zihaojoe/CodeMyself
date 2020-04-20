SHELL = /usr/bin/env bash
print: whereami.txt
	cat whereami.txt | xargs echo "I am in"

whereami.txt:
	pwd > whereami.txt

duplicate: whereami.txt
	cp whereami.txt duplicate.txt