Title: Spark + AI Summit
Icon: icon-code-outline
Date: 2018-06-06
Tags: deep learning; ai; transfer learning; tensorflow
Slug: 2018-06-06-spark-ai-summit
Summary: AI talk at the Spark+AI summit in San Francisco
Timeline: yes

https://databricks.com/session/operation-tulip-using-deep-learning-models-to-automate-auction-processes

Operation Tulip: Using Deep Learning Models to Automate Auction Processes
We are using Deep Learning models to help Royal Flora Holland automate their auction processes. With over 100,000 transactions per day and 400,000 different types of flowers and plants, Royal Flora Holland is the biggest horticulture marketplace and knowledge center in the world. An essential part of their process is having the correct photographs of the flower or plants uploaded by suppliers. These photos are uploaded daily and could have requirements.

For example, some images require a ruler to be visible or a tray to be present. Manual inspection is practically impossible. Using Keras with a Tensorflow backend we implemented a Deep Neural Network (DNN) using transfer learning for each screening criteria. We also apply heuristics and business rules. The goal is to give real-time feedback at upload time, this challenged us to run multiple deep learning models in real-enough-time.

During the journey of building the Image Detection system we have used specific implementations that can be insightful and helpful to the audience. For example, our models are not only trained in parallel but transfer learning allows us to engineer a single 1st component for all models and then having the flow distribute over each of the DNN (~90% of the work is shared among the DNNs). Our models achieve above 95% accuracy and because of the component-like architecture it’s very flexible.

Session hashtag: #AISAIS11
