# Practical Spark for Data Scientists

Spark is widely used in big data processing, feature engineering. 
It also take off in data science areas due to more machine learning algorithm integrated.

While there are quite some essential difference with tradition data processing like Pandas or SQL. 
In this article, some practical tips and essential architecture will be illustrated to enable better usage of spark for data scientists.

## Lazy execution

Executing a line of code, in tradition tools, means exactly execute that line of code.

While this is NOT the case for spark applications. Execute a line of code could means:

* The exact line of code
* From couple previous line of code until the line of the code
* From the very beginning of the application.

