# chika ![](https://github.com/moskomule/chika/workflows/pytest/badge.svg)

`chika` is a simple and easy config tool for hierarchical configurations.

## Requirements

* Python>=3.8 (`typing.get_*` requires Python3.8 or higher)

## Installation

```
pip install -U chika
```

or

```
pip install -U git+https://github.com/moskomule/chika
```


## Usage

Write typed configurations using `chika.config`, which depends on `dataclass`.

```python
# main.py
import chika

@chika.config
class ModelConfig:
    name: str = chika.choices('resnet', 'densenet')
    depth: int = 10

@chika.config
class DataConfig:
    # values that needs to be specified
    name: str = chika.required(help="name of dataset")

@chika.config
class OptimConfig:
    # 
    steps: List[int] = chika.sequence(100, 150)

@chika.config
class BaseConfig:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig

    seed: int = chika.with_help(1, help="random seed")
    use_amp: bool = False
    gpu: int = chika.choices(*range(torch.cuda.device_count()), help="id of gpu")
```

Then, wrap the main function with `chika.main(BaseConfig)`.

```python
@chika.main(BaseConfig)
def main(cfg: BaseConfig):
    set_seed(cfg.seed)
    model = ModelRegistry(cfg.model)
    ...
```

Now, `main.py` can be executed as...

```commandline
python main.py --use_amp
# cfg.use_amp == True

python main.py --model config/densenet.yaml
# cfg.model.name == densenet
# cfg.model.depth == 12

python main.py --model config/densenet.yaml --model.depth 24
# cfg.model.name == densenet
# cfg.model.depth == 24

python main.py --optim.decay_steps 120 150
# config.optim.decay_steps == [120, 150]

python main.py -h
#usage: test.py [-h] [--model MODEL] [--model.name {resnet,densenet}] [--model.depth MODEL.DEPTH] [--data DATA] --data.name DATA.NAME [--optim OPTIM] [--optim.steps OPTIM.STEPS [OPTIM.STEPS ...]]
#               [--seed SEED] [--use_amp] [--gpu {1,2,3}]
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --model MODEL         load {yaml,yml,json} file for model if necessary
#  --model.name {resnet,densenet}
#                        (default: 'resnet')
#  --model.depth MODEL.DEPTH
#                        (default: 10)
#  --data DATA           load {yaml,yml,json} file for data if necessary
#  --data.name DATA.NAME
#                        name of dataset (required) (default: None)
#  --optim OPTIM         load {yaml,yml,json} file for optim if necessary
#  --optim.steps OPTIM.STEPS [OPTIM.STEPS ...]
#                        (default: [100, 150])
#  --seed SEED           random seed (default: 1)
#  --use_amp             (default: False)
#  --gpu {1,2,3}         id of gpu (default: 1)
```

Child configs can be updated via YAML or JSON files.

```yaml
# config/densenet.yaml
model:
  name: densenet
  depth: 12 
```

For `chika.Config`, the following functions are prepared:

```python
def with_help(default, help): ...
# add help message
def choices(*values, help): ...
# add candidates that should be selected
def sequence(*values, size, help): ...
# add a list. size can be specified
def required(*, help): ...
# add a required value
def bounded(default, _from, _to, * help): ...
# add boundaries
```

### Working Directory

`change_job_dir=True` creates a unique directory for each run. 

```python
@chika.main(BaseConfig, change_job_dir=True)
def main(cfg):
    print(Path(".").resolve())
    # /home/user/outputs/0901-122412-558f5a
    print(Path(".") / "weights.pt")
    # /home/user/outputs/0901-122412-558f5a/weights.pt
    print(chika.original_path)
    # /home/user
    print(chika.resolve_original_path("weights.pt"))
    # /home/user/weights.pt
```

### Other APIs

```python
from chika import ChikaConfig
cfg = ChikaConfig.from_dict(...)

cfg.to_dict()
# {"model": {"name": "resnet", "zero_init": True, ...}, ...}
```


### Known issues and ToDos

- [ ] Configs cannot be nested twice or more than twice. `Config(Config(...))` is valid, but `Config(Config(Config(...)))` is invalid.
- [ ] Configs loaded from files are not validated.
