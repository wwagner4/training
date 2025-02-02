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

