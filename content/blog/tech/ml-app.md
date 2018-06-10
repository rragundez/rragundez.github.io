Title: Machine Learning Application Skeleton
Date: 2017-08-23
Summary: The need of the business to interact and understand the output from custom built machine learning models is increasing, here I provide an application skeleton to do just that with your Python made models.
Image: /images/blog/tech/ml-pyapp/iris_performance.png
Tags: python; flask; docker; gunicorn; front-end
Slug: ml-pyapp
Timeline:


In this blog post I provide an overview of a Python skeleton application I made. This skeleton can help you bridge the gap between your model and a machine learning application.

For example, you can use your existing Flask application, import it in `run_app.py` as `app`, and this will add the production ready features of [Gunicorn](http://gunicorn.org/).

[Take me to the code](https://github.com/rragundez/app-skeleton)

### Why bother?

The times when business saw machine learning models as black boxes with no hope of understanding are long gone.

It use to be that the data analytics or data science department of a company produced results in a silo kind of environment. Little or no interaction took place between these departments and the business side making the decisions (marketing, sales, client support, etc.). Advice coming from machine learning models consisted of reports, which were nice to have if they supported ideas from the business.

<img style="float: left;" src="/images/blog/tech/ml-pyapp/cat_peeking.jpg" width="350" hspace="20">

As data driven decisions demonstrated their value, the business side started peeking behind the curtain.

Paper/files reports have been substituted by static reporting dashboards, which themselves are being replaced by interactive ones. The business end users want to interact with the models, understand why certain predictions are made and evenmore, they want to be capable of performing predictions on the fly (imagine simultaneously having a customer on the phone and updating the probabilities of him/her buying certain products, or a marketing department tuning campaigns themselves depending on regional features).

In short, I had some time during a rainy weekend and a GDD Friday[^1], already did something similar for a client and I think it is important to bring machine learning models to the business side.

Also, as a bonus they will stop bothering you every time they need insights or a slightly different prediction.


### What's in the goody bag?

1. Template to extend a [Flask](http://flask.pocoo.org/) application using [Gunicorn](http://gunicorn.org/).

    This allows the application to be run in a more production ready environment (multiple workers and threads for example). In [here](http://docs.gunicorn.org/en/stable/settings.html) you can find a complete list of all the possible [Gunicorn](http://gunicorn.org/) settings. I added the possibility to use some of them as command line arguments. Some relevant ones are:

    - `host`
    - `port`
    - `workers` - define number of workers.
    - `threads` - number of threads on each worker.
    - `daemon` - run application in the background.
    - `access-logfile` - save access logs to a file.
    - `forwarded-allow-ips` - list allowed IP addresses.

2. Dummy application which demonstrates how to ingest several types of [user inputs](https://www.w3schools.com/html/html_form_input_types.asp) into your Python application.

    ![Dummy Application](/images/blog/tech/ml-pyapp/dummy.png)

3. Debug mode which (similar to Flask) will

    - run a single process
    - logging to debug level
    - restart process on code change
    - reload html and jinja templates on change

4. Dockerfile template to containerize the application.

    ![Docker whale](/images/blog/tech/ml-pyapp/docker.png)

5. Interactive application which runs a classifier model, outputs predictions and information about the machine learning model.

    <img src="/images/blog/tech/ml-pyapp/iris_prediction.png" width="520">
    <img src="/images/blog/tech/ml-pyapp/iris_insights.png" width="800">
    <img src="/images/blog/tech/ml-pyapp/iris_performance.png" width="580">

    The model can be run by using the UI or by directly making a post request to the endpoint.

A more complete description, a set of instructions and the code can be found in [this repository](https://github.com/rragundez/app-skeleton).

Note: I also include a `setup.py` file that you should use to install your package used in the application.

### Adios

If you structure your project following the [advice](https://blog.godatadriven.com/how-to-start-a-data-science-project-in-python) from [Henk Griffioen](https://godatadriven.com/players/henk-griffioen) (A.K.A. El Chicano), the integration of this ML application skeleton to your project should be straight forward.

<img style="float: right;" src="/images/blog/tech/ml-pyapp/dog_developer.jpg" hspace="20">

I hope this work can help you bring your models into a machine learning application, it certainly helped and will help me in the future. You can find the code [here](https://github.com/rragundez/app-skeleton).

If you have any other questions just ping me in twitter [@rragundez](https://twitter.com/rragundez).

[^1]: One Friday a month when we get to do whatever we want, it is awesome.
