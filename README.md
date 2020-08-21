# chika

`chika` is a dataclass-based simple and easy config tool

## Requirements

* Python>=3.8

## Usage

Write typed configurations using `chika.config`.

```python
# main.py
import chika

@chika.config
class Config:
    model: ModelConfig
    data: DataConfig

    seed: int = 1
    use_amp: bool = False

@chika.main(config_file="config.yaml")
def main(cfg: Config):
    model = ModelRegistry(cfg.model)
    ...

# or

def main2():
    cfg = Config(model=ModelConfig(), 
                 ...)
```

```yaml
# config.yaml
model:
  name: resnet
  ... 
```

```commandline
python main.py --use_amp
# cfg.use_amp == True

python main.py model=config/densenet.yaml
# cfg.model == densenet

python main.py --model.name resnet
# cfg.model.name == resnet
```

### Other APIs

```python
>>> cfg.show()
# model: name=resnet
#        zero_init=True
#        ...
# ...

>>> cfg.to_dict()
# {"model": {"name": "resnet", "zero_init": True, ...}, ...}

>>> cfg.load_file("config.yaml")
>>> cfg.load_file("config.json")
>>> cfg.load_args()

>>> chika.unique_path
# Path("outputs/202008200001-609938")
```