## Training for sumosim robots

### Getting started

#### Prerequisits
* python 3.10+
* poetry. For creating a virtual environment
* docker and docker compose

Create the virual environment by running

```
poetry install
poetry shell
```
#### Starting a sumosim robo map

##### Start a database
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


### Formating
in project root call: `ruff format training/`

### Linting
in project root call: `ruff check` or `ruff check --fix`
