## Training for sumosim robots

### Getting started

#### Prerequisits
* uv
* docker and docker compose

##### Start a database

In SUMOSIM/sumosim run: 
```
docker compose up -d
```

##### Start a sumosim simulator

For details see (sumosim simulator)[https://github.com/SUMOSIM1/sumosim/tree/reinforcement]

```
sumosim start --port [PORTNUMBER]

e.g

sumosim start --port 5555
```
### Running
`uv run sumot --help`

### Formating
in project root call: `ruff format`

### Linting
in project root call: `ruff check` or `ruff check --fix`

### Create a video from images
`ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p q-values-002.mp4`

### Docker 

TODO Docker

 * config host and port for database. db_host, db_port . host -> sim_host, sim_port
 * use local user in docker sumot. 
  * create necesaaey directories during build and set to 775

```
docker network create sumo01
docker run -d --rm --name sumo1 --network sumot01 sumo sumo udp --port 4401
docker run  -e PYTHONUNBUFFERED=True --network sumot01 \
-v $HOME/tmp/sumosim/q/docker:/root/tmp/sumosim/q \
sumot uv run sumot qtrain -n D01 --auto-naming -e 500 -p 4401 -h sumo1



docker network create sumo02
docker run -d --rm --name sumo2 --network sumot02 sumo sumo udp --port 4402

docker run -e PYTHONUNBUFFERED=True --network sumot02 --user $(id -u):$(id -g) -v $(echo $HOME)/tmp:/tmp sumot uv run sumot qtrain -n D04 --auto-naming -e 20 -p 4402 -h sumo2 -o /tmp/sumosim/q/docker

docker container ls -a -f "NAME=sumo-train" -q | xargs docker container rm
```


# Concept CV values


```
variable

class CvValue:
	id: str,
	learning_rate: float,
	epsilon: float,
	discount_factor: float,
	
class CvValuesName(Enum):
	FIRST = "first"


class CvValues:

  def(self, index) -> CvValue 



                    0        1     2     3      4  
learning_rate    L  0.01     0.1,  0.01, 0.001, 0.0005 
epsilon          E  0.05     0.1,  0.05, 0.005, 0.001 
discount_factor  D  0.95     0.99, 0.95, 0.5,   0.25



n = 125

parallel_count = 25 paralelle läufe

parallel_numbers = list[int] 

parallel_numbers_1 = [ 0,  9] range(0, 10) für work computer
parallel_numbers_2 = [10, 14] range(10, 15) für ben
```

