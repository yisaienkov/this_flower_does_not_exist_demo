# This Flower Does Not Exist 🌸


## 1. Start app

### 1.1. Build the docker image

```bash
$ docker build -t this_flower_does_not_exist_demo .
```

### 1.2. Run the docker container

```bash
$ docker run -e PORT=5001 -p 5001:5001 this_flower_does_not_exist_demo
```