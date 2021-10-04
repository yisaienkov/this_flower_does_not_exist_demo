# This Flower Does Not Exist 🌸

Try the app: [thisflowerdoesnotexist.herokuapp.com](https://thisflowerdoesnotexist.herokuapp.com) (for the first time, it can boot ~30 sec).


## Interface example

![image](https://i.imgur.com/ldXSL9O.png)


- Select the type of flowers to generate.
- Change the truncation threshold to change between images fidelity and diversity. 
- Set the different sizes of the grid.


## Local start

### Build the docker image

```bash
$ docker build -t this_flower_does_not_exist_demo .
```

### Run the docker container

```bash
$ docker run -e PORT=5001 -p 5001:5001 this_flower_does_not_exist_demo
```