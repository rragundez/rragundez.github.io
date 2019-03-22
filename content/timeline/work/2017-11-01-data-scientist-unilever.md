Title: Senior Data Scientist @ Unilever
Icon: icon-code-outline
Date: 2017-11-01
Tags: data science; Spark; Python; C++; Cython; Jupyter; TF-IDF; n-grams; Azure; ssh; git
Slug: 2017-11-01-data-scientist-unilever
Summary: As a consultant from GoDataDriven working as a senior data scientist for Unilever.
Timeline: yes

I worked developing a fuzzy name matching algorithm for entity data of all countries within the Unilever market. Data coming from different sources has been unavoidably duplicated. The goal was to create golden records which will be enriched from all matching ones. This is a known problem since already with 1 million records yields a cartesian product of 10^12 comparisons, which is unfeasible. I was able to produce a high performance approach which mixed machine learning, distributed computing and highly optimized algorithms, this approach yield a run-time of 1.5hrs for ~10 million records across 50 countries. The algorithm was put in place in production by data engineers.
