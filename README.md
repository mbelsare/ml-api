# ml-api
Deployment of ML Model as an API Endpoint

Prerequisites
====
* python3
* Docker desktop

Packages used
====
* `flask` - web framework for building web services
* `tensorflow` - for loading and running model files
* `ResNet50` - to instantiate the ResNet50 architecture & create a pretrained model

##### **tf.keras.applications.resnet50** module has 2 additional functions

* `decode_predictions()`: Decodes the prediction of an ImageNet model.
* `preprocess_input()`: Preprocesses a tensor or Numpy array encoding a batch of images.

System flow
====
![API overview](images/MLApi.png)

Building the API
=====
* The app is packaged as a docker image with the `Dockerfile` provided in the repo
```
docker build -t mlapi .
```

Running the ML API
====
```
docker run -p 8002:5000 mlapi
```
* Once the docker container (web server) is up and running at default port 5000, navigate to POSTMAN and run the below curl commands

#### Health Check (GET)
```
curl --location 'http://localhost:8002/'
```

#### Get the prediction (POST)
```
curl --location 'http://localhost:8002/predict' \
--header 'Content-Type: application/json' \
--data '
{
    "img_url" : "https://cdn.pixabay.com/photo/2017/10/04/09/56/laboratory-2815641_1280.jpg"
}'
```
* The POST request takes in a sample image URL as request body, passes the image to the model and returns a probable score for different objects in the image

##### Sample output
```
[
    {
        "label": "lab_coat",
        "probability": 0.9986553192138672
    },
    {
        "label": "beaker",
        "probability": 0.0005878391093574464
    },
    {
        "label": "perfume",
        "probability": 0.0003308393061161041
    }
]
```
