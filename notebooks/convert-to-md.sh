#!/usr/bin/env bash

echo "usage: convert jupyter notebook to markdown format"
echo "    cd notebooks"
echo "    ./convert-to-md.sh 2018-03-05-Supervised-Learning-Explained-1"

set -x

name=$1

short_name=${name:11}

title="${short_name//-/ }"

jupyter nbconvert --to markdown "${name}.ipynb"

echo "---" > "${name}.md.tmp"
echo "layout: post" >> "${name}.md.tmp"
echo "title: $title" >> "${name}.md.tmp"
echo "comments: true" >> "${name}.md.tmp"
echo "tags: jupyter, machine learning, big data" >> "${name}.md.tmp"
echo "---" >> "${name}.md.tmp"


cat -s "${name}.md" >> "${name}.md.tmp"

sed -i "s|${name}_files|{{ site.baseurl }}/notebooks/${name}_files|g" "${name}.md.tmp"
sed -i "s|readmore|<!-- readmore -->|g" "${name}.md.tmp"
mv "${name}.md.tmp" "../_posts/${name}.md"

rm "${name}.md"
