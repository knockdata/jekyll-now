---
layout: post
title: Change Jekyll Default excerpt_separator
tags: jekyll 
---

Jekyll will take first part of a blog post and display it. 
The break point is called excerpt_separator, 


The default excerpt_separator is "\n\n", see [jekyll configuration](refer https://jekyllrb.com/docs/configuration/). 
The ruby file provide this configure is in [configuration](https://github.com/jekyll/jekyll/blob/master/lib/jekyll/configuration.rb)


While using "\n\n", posts will be most likely break at very early part. Due to heavy usage or new lines in markdown.

Fortunately we can configured it in file `_config.yml` like 


	excerpt_separator: "<!-- readmore -->"

So that in posts, we can specify the break point manually by type in `<!-- readmore -->` manually

> NOTE: The quote is important when configure it. Use like \n\n\n will have no effect.