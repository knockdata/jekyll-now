#!/bin/bash

docker run -it --rm -p 8000:8888 -v `pwd`:/opt/knockdata knockdata/python3 jupyter notebook --notebook-dir=/opt/knockdata --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token=''
